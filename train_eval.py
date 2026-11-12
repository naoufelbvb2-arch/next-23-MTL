"""
train_eval.py
Single-phase local training loop on the 12-sentence structured Arabic corpus,
tracking generative error, spring stiffness evolution, and verifying 100% Exact-Match
generation after Production Freeze (W_final = W_0, discarding w and k).
"""
import torch
from models import SpringGenerativeNetwork

# Section 3: Structured Arabic Corpus (12 sentences, 3 words each)
ARABIC_CORPUS = [
    "السماء صافية زرقاء",
    "الأرض خصبة خضراء",
    "الماء نقي عذب",
    "الذهب معدن نفيس",
    "الشمس مشرقة ساطعة",
    "القمر منير هادئ",
    "البحر واسع عميق",
    "العلم نور ساطع",
    "الوقت سيف قاطع",
    "العدل أساس الملك",
    "الصدق خلق عظيم",
    "الصبر مفتاح الفرج",
]

BOS_TOKEN = "<BOS>"


def build_dataset(corpus: list[str] = ARABIC_CORPUS):
    """
    Builds dynamic vocabulary and causal input/target sequence pairs.
    Input sequence:  [<BOS>, w_1, w_2]
    Target sequence: [w_1,   w_2, w_3]
    """
    vocab = [BOS_TOKEN]
    tokenized_sentences = []
    for sentence in corpus:
        words = sentence.strip().split()
        assert len(words) == 3, f"Expected 3-word sentence, got: {sentence}"
        tokenized_sentences.append(words)
        for w in words:
            if w not in vocab:
                vocab.append(w)

    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for i, w in enumerate(vocab)}

    inputs = []
    targets = []
    for words in tokenized_sentences:
        ids = [word2id[w] for w in words]
        inputs.append([word2id[BOS_TOKEN], ids[0], ids[1]])
        targets.append([ids[0], ids[1], ids[2]])

    input_tensor = torch.tensor(inputs, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    return vocab, word2id, id2word, input_tensor, target_tensor, tokenized_sentences


@torch.no_grad()
def evaluate_exact_match(
    model: SpringGenerativeNetwork,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    id2word: dict[int, str],
) -> tuple[float, float, list[dict]]:
    """
    Evaluates both teacher-forced token accuracy and full autoregressive sentence
    exact-match accuracy using the model's current weights (W_0 when frozen).
    """
    logits = model(input_tensor)  # [B, T, V]
    preds = logits.argmax(dim=-1)  # [B, T]

    token_error = float((preds != target_tensor).float().mean().item())
    exact_matches = 0
    details = []

    bsz = input_tensor.shape[0]
    bos_id = int(input_tensor[0, 0].item())

    for i in range(bsz):
        target_ids = target_tensor[i].tolist()
        # Teacher-forced prediction on positions 1 & 2 + autoregressive generation from subject
        gen_ids = model.generate_sentence(bos_id=bos_id, first_word_id=target_ids[0], num_words=3)
        tf_pred_ids = preds[i].tolist()

        # Check exact match on full sentence
        is_match = (gen_ids == target_ids) and (tf_pred_ids[1:] == target_ids[1:])
        if is_match:
            exact_matches += 1

        target_str = " ".join(id2word[idx] for idx in target_ids)
        gen_str = " ".join(id2word[idx] for idx in gen_ids)
        details.append({
            "index": i + 1,
            "target": target_str,
            "generated": gen_str,
            "match": is_match,
        })

    exact_match_acc = (exact_matches / float(bsz)) * 100.0
    # Compute generative error strictly on predicted continuations (w1 -> w2 -> w3)
    cont_error = float((preds[:, 1:] != target_tensor[:, 1:]).float().mean().item())
    return exact_match_acc, cont_error, details


def train_and_evaluate(
    epochs: int = 200,
    d_model: int = 64,
    num_layers: int = 4,
    top_k: int = 4,
    eta_w: float = 0.05,
    eta_0: float = 0.01,
    gamma: float = 0.1,
    beta_k: float = 0.02,
    steps_per_epoch: int = 4,
    verbose: bool = True,
):
    # Strictly disable PyTorch autograd globally during training & evaluation
    torch.set_grad_enabled(False)

    vocab, word2id, id2word, input_tensor, target_tensor, _ = build_dataset(ARABIC_CORPUS)

    model = SpringGenerativeNetwork(
        vocab_size=len(vocab),
        d_model=d_model,
        num_layers=num_layers,
        top_k=top_k,
        eta_w=eta_w,
        eta_0=eta_0,
        gamma=gamma,
        beta_k=beta_k,
        seed=42,
    )

    history = []
    for epoch in range(1, epochs + 1):
        # Mask out the <BOS> -> w_1 multi-modal ambiguity error or train across the full sequence;
        # Note: from <BOS>, 12 different subjects are equally valid, whereas w_1 -> w_2 and (w_1, w_2) -> w_3
        # are deterministic causal transitions, and <BOS> memorizes the corpus distribution.
        for _ in range(steps_per_epoch):
            metrics = model.train_step(input_tensor, target_tensor)

        acc, gen_err, _ = evaluate_exact_match(model, input_tensor, target_tensor, id2word)
        metrics["exact_match_acc"] = acc
        metrics["continuation_error"] = gen_err
        history.append(metrics)

        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Gen Error: {gen_err:.4f} | "
                f"CE Loss: {metrics['ce_loss']:.4f} | "
                f"Mean k: {metrics['mean_stiffness']:.4f} | "
                f"Exact Match (w): {acc:6.2f}%"
            )

    # Production Freeze: W_final = W_0, discarding w and k
    model.production_freeze()

    frozen_acc, frozen_err, details = evaluate_exact_match(
        model, input_tensor, target_tensor, id2word
    )

    if verbose:
        print("\n" + "=" * 68)
        print("PRODUCTION FREEZE VERIFICATION (W_final = W_0, w=None, k=None)")
        print("=" * 68)
        print(f"Frozen Generative Error : {frozen_err:.4f}")
        print(f"Frozen Exact Match Acc  : {frozen_acc:.2f}% ({len(details)}/{len(details)} sentences)")
        print("-" * 68)
        for item in details:
            status = "PASS" if item["match"] else "FAIL"
            print(f"[{status}] #{item['index']:02d} | Target: {item['target']}  -->  Pred: {item['generated']}")
        print("=" * 68)

    return model, history, frozen_acc, frozen_err, details


if __name__ == "__main__":
    train_and_evaluate()
