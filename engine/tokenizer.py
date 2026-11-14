"""
engine/tokenizer.py
Fast, deterministic hybrid Word + Subword BPE English Tokenizer for SS-ENN.
Supports special tokens (<pad>, <bos>, <eos>, <unk>), subword fallback for rare/unseen
morphological variants, and vectorized batch encoding/decoding.
Strictly zero autograd / zero backward().
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


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

    def to_dict(self) -> dict:
        """Serializes tokenizer configuration and vocabulary cleanly to JSON dict."""
        ordered_vocab = [
            self.id_to_token[idx] for idx in sorted(self.id_to_token.keys())
        ]
        return {
            "format_version": "ssenn_tokenizer_1.0",
            "max_vocab_size": int(self.max_vocab_size),
            "vocab": ordered_vocab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnglishSubwordTokenizer":
        """Restores tokenizer deterministically from a serialized dictionary."""
        tok = cls(max_vocab_size=int(data["max_vocab_size"]))
        tok.token_to_id.clear()
        tok.id_to_token.clear()
        tok.word_cache.clear()
        for idx, token in enumerate(data["vocab"]):
            tok.token_to_id[token] = idx
            tok.id_to_token[idx] = token
        return tok

    def save(self, path: str | Path) -> Path:
        """Saves tokenizer vocabulary deterministically to `tokenizer.json`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EnglishSubwordTokenizer":
        """Loads tokenizer cleanly from a saved `tokenizer.json` file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _format_bpe_piece(raw: bytes, token_id: int, existing: dict[str, int]) -> str:
    """Formats a byte sequence into a unique human-readable BPE vocabulary token."""
    try:
        decoded = raw.decode("utf-8")
        pretty = (
            decoded.replace(" ", "Ġ")
            .replace("\n", "Ċ")
            .replace("\r", "č")
            .replace("\t", "ĉ")
        )
        # Ensure printable non-empty token string without control chars or collisions
        if pretty and all(ch.isprintable() for ch in pretty) and pretty not in existing:
            return pretty
    except UnicodeDecodeError:
        pass
    hex_name = "".join(f"<0x{b:02X}>" for b in raw)
    if hex_name not in existing:
        return hex_name
    return f"<bpe_{token_id}>"


class MultilingualByteBPETokenizer:
    """Lossless Byte-Level BPE tokenizer supporting Arabic, French, English, emojis, and code.

    Properties:
    - Special tokens: `<pad>` (0), `<bos>` (1), `<eos>` (2), `<unk>` (3)
    - Base UTF-8 byte tokens: `<0x00>` .. `<0xFF>` at IDs `4 .. 259` (0 `<unk>` guarantee)
    - Unicode NFC normalization + lossless whitespace/diacritic-preserving pre-tokenization
    - Fast frequency-weighted BPE training (`fit`) and cached encoding (`encode`)
    """

    FORMAT_VERSION = "ssenn_multilingual_bpe_1.0"
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    BYTE_OFFSET = 4

    def __init__(self, max_vocab_size: int = 8192):
        max_v = int(max_vocab_size)
        if max_v < self.BYTE_OFFSET + 256:
            raise ValueError(
                f"MultilingualByteBPETokenizer requires max_vocab_size >= {self.BYTE_OFFSET + 256}, got {max_v}."
            )
        self.max_vocab_size = max_v
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self._id_to_bytes: list[bytes] = []
        self._merges: list[tuple[int, int, int]] = []
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._chunk_cache: dict[str, list[int]] = {}
        self._init_base_vocab()

    def _init_base_vocab(self) -> None:
        self.token_to_id = {}
        self.id_to_token = []
        self._id_to_bytes = []
        self._merges = []
        self._merge_ranks = {}
        self._chunk_cache.clear()
        for tok in self.SPECIAL_TOKENS:
            tid = len(self.id_to_token)
            self.token_to_id[tok] = tid
            self.id_to_token.append(tok)
            self._id_to_bytes.append(b"")
        for b in range(256):
            tid = len(self.id_to_token)
            tok = f"<0x{b:02X}>"
            self.token_to_id[tok] = tid
            self.id_to_token.append(tok)
            self._id_to_bytes.append(bytes([b]))

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @staticmethod
    def _char_kind(ch: str) -> str:
        cat = unicodedata.category(ch)
        major = cat[0]
        if major in ("L", "M"):
            return "W"  # Letters + combining diacritics (Arabic tashkeel, French accents)
        if major == "N":
            return "N"
        if ch in ("\n", "\r"):
            return "NL"
        if ch in (" ", "\t") or major == "Z":
            return "SP"
        return "P"

    @classmethod
    def _pretokenize(cls, text: str) -> list[str]:
        """Lossless Unicode NFC pre-tokenizer preserving spaces, tabs, newlines, and marks."""
        norm = unicodedata.normalize("NFC", text)
        n = len(norm)
        if n == 0:
            return []

        chunks: list[str] = []
        i = 0
        while i < n:
            ch = norm[i]
            kind = cls._char_kind(ch)
            if kind in ("W", "N"):
                j = i + 1
                while j < n and cls._char_kind(norm[j]) == kind:
                    j += 1
                chunks.append(norm[i:j])
                i = j
            elif kind == "NL":
                if ch == "\r" and i + 1 < n and norm[i + 1] == "\n":
                    chunks.append("\r\n")
                    i += 2
                else:
                    j = i + 1
                    while j < n and norm[j] == ch:
                        j += 1
                    chunks.append(norm[i:j])
                    i = j
            elif kind == "SP":
                # Attach a single ASCII space directly to the following word/digit/punct chunk
                if ch == " " and i + 1 < n:
                    next_kind = cls._char_kind(norm[i + 1])
                    if next_kind in ("W", "N"):
                        j = i + 2
                        while j < n and cls._char_kind(norm[j]) == next_kind:
                            j += 1
                        chunks.append(norm[i:j])
                        i = j
                        continue
                    if next_kind == "P":
                        chunks.append(norm[i : i + 2])
                        i += 2
                        continue
                j = i + 1
                while j < n and norm[j] == ch and (j - i) < 16:
                    # Leave final single space before a word/digit/punct to attach next turn
                    if (
                        ch == " "
                        and (j + 1) < n
                        and norm[j + 1] != " "
                        and cls._char_kind(norm[j + 1]) in ("W", "N", "P")
                    ):
                        break
                    j += 1
                chunks.append(norm[i:j])
                i = j
            else:
                # Punctuation/symbol: keep contiguous identical symbol runs up to 4 chars (e.g. ==, ---)
                j = i + 1
                while j < n and norm[j] == ch and (j - i) < 4:
                    j += 1
                chunks.append(norm[i:j])
                i = j
        return chunks

    @staticmethod
    def _merge_pair_in_seq(seq: list[int], a: int, b: int, new_id: int) -> list[int]:
        out: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def fit(
        self,
        texts: list[str],
        pad_to_max_vocab: bool = True,
        min_frequency: int = 2,
    ) -> "MultilingualByteBPETokenizer":
        """Learns byte-level BPE merges from a multilingual corpus up to `max_vocab_size`."""
        self._init_base_vocab()
        chunk_counter: Counter[str] = Counter()
        for text in texts:
            for chunk in self._pretokenize(text):
                chunk_counter[chunk] += 1

        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, freq in chunk_counter.items():
            raw = chunk.encode("utf-8")
            if len(raw) >= 2:
                words.append([self.BYTE_OFFSET + b for b in raw])
                freqs.append(int(freq))

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_indices: dict[tuple[int, int], set[int]] = {}
        for idx, seq in enumerate(words):
            f = freqs[idx]
            for k in range(len(seq) - 1):
                pair = (seq[k], seq[k + 1])
                pair_counts[pair] += f
                s = pair_indices.get(pair)
                if s is None:
                    pair_indices[pair] = {idx}
                else:
                    s.add(idx)

        target_merges = self.max_vocab_size - len(self.id_to_token)
        min_freq = max(1, int(min_frequency))

        for _ in range(max(0, target_merges)):
            # Prune zero/stale counts while finding the deterministic maximum-frequency pair
            best_pair = None
            best_score = (min_freq - 1, 0, 0)
            stale_pairs: list[tuple[int, int]] = []
            for pair, count in pair_counts.items():
                if count <= 0:
                    stale_pairs.append(pair)
                    continue
                if count >= min_freq:
                    score = (count, -pair[0], -pair[1])
                    if score > best_score:
                        best_score = score
                        best_pair = pair
            for sp in stale_pairs:
                pair_counts.pop(sp, None)
                pair_indices.pop(sp, None)

            if best_pair is None:
                break

            a, b = best_pair
            new_id = len(self.id_to_token)
            merged_bytes = self._id_to_bytes[a] + self._id_to_bytes[b]
            tok_str = _format_bpe_piece(merged_bytes, new_id, self.token_to_id)
            self.token_to_id[tok_str] = new_id
            self.id_to_token.append(tok_str)
            self._id_to_bytes.append(merged_bytes)
            rank = len(self._merges)
            self._merges.append((a, b, new_id))
            self._merge_ranks[best_pair] = rank

            affected = list(pair_indices.pop(best_pair, ()))
            pair_counts.pop(best_pair, None)

            for idx in affected:
                old_seq = words[idx]
                if len(old_seq) < 2:
                    continue
                f = freqs[idx]
                # Remove old pairs for this chunk
                for k in range(len(old_seq) - 1):
                    p_old = (old_seq[k], old_seq[k + 1])
                    if p_old != best_pair and p_old in pair_counts:
                        pair_counts[p_old] -= f
                        if pair_counts[p_old] <= 0:
                            pair_counts.pop(p_old, None)
                            pair_indices.pop(p_old, None)
                        else:
                            pset = pair_indices.get(p_old)
                            if pset is not None:
                                pset.discard(idx)

                new_seq = self._merge_pair_in_seq(old_seq, a, b, new_id)
                words[idx] = new_seq
                # Add updated pairs
                if len(new_seq) >= 2:
                    for k in range(len(new_seq) - 1):
                        p_new = (new_seq[k], new_seq[k + 1])
                        pair_counts[p_new] += f
                        pset = pair_indices.get(p_new)
                        if pset is None:
                            pair_indices[p_new] = {idx}
                        else:
                            pset.add(idx)

        if pad_to_max_vocab:
            unused_idx = 0
            while len(self.id_to_token) < self.max_vocab_size:
                pad_tok = f"<unused_{unused_idx}>"
                unused_idx += 1
                if pad_tok not in self.token_to_id:
                    tid = len(self.id_to_token)
                    self.token_to_id[pad_tok] = tid
                    self.id_to_token.append(pad_tok)
                    self._id_to_bytes.append(b"")

        return self

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached

        ids = [self.BYTE_OFFSET + b for b in chunk.encode("utf-8")]
        if len(ids) >= 2 and self._merge_ranks:
            merge_ranks = self._merge_ranks
            while len(ids) >= 2:
                best_pair = None
                best_rank = None
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    r = merge_ranks.get(pair)
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank = r
                        best_pair = pair
                if best_pair is None:
                    break
                a, b = best_pair
                new_id = self._merges[best_rank][2]
                ids = self._merge_pair_in_seq(ids, a, b, new_id)

        if len(self._chunk_cache) >= 65536:
            self._chunk_cache.clear()
        self._chunk_cache[chunk] = ids
        return ids

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in self._pretokenize(text):
            ids.extend(self._encode_chunk(chunk))
            if max_length is not None and len(ids) >= max_length:
                break
        if add_eos:
            if max_length is None or len(ids) < max_length:
                ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs via byte concatenation so split UTF-8 characters remain 100% lossless."""
        out_parts: list[str] = []
        byte_buf = bytearray()
        vocab_len = len(self._id_to_bytes)
        special_set = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}

        for tid in token_ids:
            tid_i = int(tid)
            if 0 <= tid_i < vocab_len:
                raw = self._id_to_bytes[tid_i]
                if raw:
                    byte_buf.extend(raw)
                elif not skip_special_tokens and tid_i in special_set:
                    if byte_buf:
                        out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                        byte_buf.clear()
                    out_parts.append(self.id_to_token[tid_i])
            elif not skip_special_tokens:
                if byte_buf:
                    out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                    byte_buf.clear()
                out_parts.append("<unk>")

        if byte_buf:
            out_parts.append(bytes(byte_buf).decode("utf-8", errors="replace"))
        return "".join(out_parts)

    def encode_batch(
        self,
        texts: list[str],
        seq_len: int,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[list[int]]:
        batch: list[list[int]] = []
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos, max_length=seq_len)
            if len(ids) < seq_len:
                ids = ids + [self.pad_id] * (seq_len - len(ids))
            batch.append(ids)
        return batch

    def to_dict(self) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "max_vocab_size": self.max_vocab_size,
            "vocab": list(self.id_to_token),
            "merges": [[int(a), int(b), int(new_id)] for a, b, new_id in self._merges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultilingualByteBPETokenizer":
        tok = cls(max_vocab_size=int(data["max_vocab_size"]))
        vocab = [str(v) for v in data["vocab"]]
        merges_raw = data.get("merges", [])

        # Reset dynamic state past the 260 fixed tokens
        tok._init_base_vocab()
        merges: list[tuple[int, int, int]] = [(int(a), int(b), int(nid)) for a, b, nid in merges_raw]
        merge_by_id = {nid: (a, b, rank) for rank, (a, b, nid) in enumerate(merges)}

        base_len = len(tok.id_to_token)
        for tid in range(base_len, len(vocab)):
            tok_str = vocab[tid]
            tok.id_to_token.append(tok_str)
            tok.token_to_id[tok_str] = tid
            if tid in merge_by_id:
                a, b, _ = merge_by_id[tid]
                tok._id_to_bytes.append(tok._id_to_bytes[a] + tok._id_to_bytes[b])
            else:
                tok._id_to_bytes.append(b"")

        tok._merges = merges
        tok._merge_ranks = {(a, b): rank for rank, (a, b, _nid) in enumerate(merges)}
        tok._chunk_cache.clear()
        return tok

    def save(self, path: str | Path) -> Path:
        """Saves multilingual byte BPE vocabulary and merges deterministically to `tokenizer.json`."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "MultilingualByteBPETokenizer":
        """Loads MultilingualByteBPETokenizer from a saved `tokenizer.json` file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def load_tokenizer(path: str | Path) -> EnglishSubwordTokenizer | MultilingualByteBPETokenizer:
    """Auto-detects and loads either `MultilingualByteBPETokenizer` or `EnglishSubwordTokenizer`."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("format_version") == MultilingualByteBPETokenizer.FORMAT_VERSION:
        return MultilingualByteBPETokenizer.from_dict(data)
    return EnglishSubwordTokenizer.from_dict(data)
