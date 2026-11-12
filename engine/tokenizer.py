"""
engine/tokenizer.py
Fast, deterministic hybrid Word + Subword BPE English Tokenizer for SS-ENN.
Supports special tokens (<pad>, <bos>, <eos>, <unk>), subword fallback for rare/unseen
morphological variants, and vectorized batch encoding/decoding.
Strictly zero autograd / zero backward().
"""
import re
from collections import Counter


class EnglishSubwordTokenizer:
    """
    Hybrid Word + Subword BPE Tokenizer tailored for TinyStories-style English corpora.
    Splits words and punctuation cleanly, retains high-frequency whole words as single tokens,
    and decomposes rare/compound words into common English stems/suffixes (-ing, -ed, -ly, -er, -s)
    or character bigrams so `<unk>` is minimized and compositional generalization is supported.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    COMMON_SUFFIXES = ("ing", "ed", "ly", "er", "est", "ness", "ful", "less", "ment", "es", "s")

    # Regex that separates words and punctuation while preserving contractions
    _TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,!?;:\-\"()]")

    def __init__(self, max_vocab_size: int = 2048):
        self.max_vocab_size = max_vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.word_cache: dict[str, list[int]] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.token_to_id.clear()
        self.id_to_token.clear()
        for idx, tok in enumerate(self.SPECIAL_TOKENS):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        if len(self.token_to_id) >= self.max_vocab_size:
            return self.unk_id
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    def pre_tokenize(self, text: str) -> list[str]:
        return self._TOKEN_PATTERN.findall(text.lower())

    def fit(self, texts: list[str], pad_to_max_vocab: bool = True) -> "EnglishSubwordTokenizer":
        """
        Builds vocabulary from a list of English texts:
        1. Registers special tokens + ASCII characters + common English subword suffixes/stems.
        2. Registers most frequent whole words.
        3. Optionally pads with synthetic subword bigrams/trigrams up to `max_vocab_size`
           so preset parameter counts (e.g. V=2048 for 1.5M preset) are exact.
        """
        self._init_base_vocab()
        self.word_cache.clear()

        # 1. Add single characters and punctuation
        for ch in "abcdefghijklmnopqrstuvwxyz0123456789.,!?;:-'\"()":
            self._add_token(ch)

        # 2. Add common English subword suffixes (prefixed with '##' to mark continuation)
        for suf in self.COMMON_SUFFIXES:
            self._add_token(f"##{suf}")

        # 3. Count word frequencies across corpus
        word_counts: Counter = Counter()
        subword_counts: Counter = Counter()

        for text in texts:
            words = self.pre_tokenize(text)
            word_counts.update(words)
            for w in words:
                for suf in self.COMMON_SUFFIXES:
                    if len(w) > len(suf) + 2 and w.endswith(suf):
                        stem = w[: -len(suf)]
                        subword_counts[stem] += 1
                        break

        # 4. Add frequent whole words first
        for word, _ in word_counts.most_common():
            if len(self.token_to_id) >= self.max_vocab_size:
                break
            self._add_token(word)

        # 5. Add stems and character n-grams
        for stem, _ in subword_counts.most_common():
            if len(self.token_to_id) >= self.max_vocab_size:
                break
            self._add_token(stem)

        if pad_to_max_vocab and len(self.token_to_id) < self.max_vocab_size:
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            for c1 in alphabet:
                for c2 in alphabet:
                    if len(self.token_to_id) >= self.max_vocab_size:
                        break
                    self._add_token(f"##{c1}{c2}")
                if len(self.token_to_id) >= self.max_vocab_size:
                    break
            idx = 0
            while len(self.token_to_id) < self.max_vocab_size:
                self._add_token(f"<sub_{idx}>")
                idx += 1

        return self

    def _encode_word(self, word: str) -> list[int]:
        if word in self.word_cache:
            return self.word_cache[word]

        if word in self.token_to_id:
            res = [self.token_to_id[word]]
            self.word_cache[word] = res
            return res

        # Try stem + ##suffix decomposition
        for suf in self.COMMON_SUFFIXES:
            suf_tok = f"##{suf}"
            if len(word) > len(suf) + 1 and word.endswith(suf) and suf_tok in self.token_to_id:
                stem = word[: -len(suf)]
                if stem in self.token_to_id:
                    res = [self.token_to_id[stem], self.token_to_id[suf_tok]]
                    self.word_cache[word] = res
                    return res

        # Fallback to character/subword pieces
        ids = []
        i = 0
        while i < len(word):
            if i > 0 and i + 2 <= len(word):
                bg = f"##{word[i:i+2]}"
                if bg in self.token_to_id:
                    ids.append(self.token_to_id[bg])
                    i += 2
                    continue
            ch = word[i]
            ids.append(self.token_to_id.get(ch, self.unk_id))
            i += 1

        self.word_cache[word] = ids
        return ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        words = self.pre_tokenize(text)
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for w in words:
            ids.extend(self._encode_word(w))
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces: list[str] = []
        special_set = set(self.SPECIAL_TOKENS)
        punct_set = {".", ",", "!", "?", ";", ":", ")", "-"}

        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), self.UNK_TOKEN)
            if skip_special_tokens and tok in special_set:
                continue
            if tok.startswith("##"):
                if pieces:
                    pieces[-1] = pieces[-1] + tok[2:]
                else:
                    pieces.append(tok[2:])
            elif tok in punct_set and pieces:
                pieces[-1] = pieces[-1] + tok
            else:
                pieces.append(tok)

        text = " ".join(pieces)
        if text:
            text = text[0].upper() + text[1:]
        return text

    def encode_batch(
        self,
        texts: list[str],
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        """
        Encodes a list of stories into fixed-length causal token windows of length `seq_len`.
        """
        windows: list[list[int]] = []
        for t in texts:
            ids = self.encode(t, add_bos=add_bos, add_eos=add_eos)
            if len(ids) < seq_len:
                ids = ids + [self.pad_id] * (seq_len - len(ids))
                windows.append(ids)
            else:
                # Extract non-overlapping or strided windows of length seq_len
                for start in range(0, len(ids) - seq_len + 1, max(1, seq_len - 4)):
                    windows.append(ids[start : start + seq_len])
        return windows
