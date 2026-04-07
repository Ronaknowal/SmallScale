"""Byte-Pair Encoding tokenizer built from scratch.

Building your own tokenizer demonstrates understanding of a critical but
often-overlooked component. Vocab size is an important ablation dimension:
smaller vocab = less memory but higher sequence length (more compute),
larger vocab = more embedding params but shorter sequences.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Optional


# GPT-4 style pre-tokenization regex (splits on whitespace, punctuation, numbers)
PRE_TOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+"""
)


class BPETokenizer:
    """Minimal BPE tokenizer trained from scratch.

    Steps:
    1. Pre-tokenize text into words
    2. Initialize vocab with individual bytes (256 base tokens)
    3. Iteratively merge most frequent byte pairs
    4. Build final vocab of desired size

    Special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self):
        self.merges = {}  # (token_a, token_b) -> merged_token_id
        self.vocab = {}  # token_id -> bytes
        self.inverse_vocab = {}  # bytes -> token_id

    def train(self, texts: List[str], vocab_size: int = 32000, verbose: bool = True):
        """Train BPE on a list of text strings."""
        assert vocab_size > 256 + len(self.SPECIAL_TOKENS)

        # Start with byte-level vocabulary + special tokens
        num_special = len(self.SPECIAL_TOKENS)
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Shift byte tokens to make room for special tokens at the start
        self.vocab = {}
        for tok, idx in self.SPECIAL_TOKENS.items():
            self.vocab[idx] = tok.encode("utf-8")

        for i in range(256):
            self.vocab[i + num_special] = bytes([i])

        # Pre-tokenize and convert to byte sequences
        word_freqs = Counter()
        for text in texts:
            words = PRE_TOKENIZE_PATTERN.findall(text)
            for word in words:
                # Convert word to tuple of byte token ids
                byte_ids = tuple(b + num_special for b in word.encode("utf-8"))
                word_freqs[byte_ids] += 1

        # Iteratively merge most frequent pairs
        num_merges = vocab_size - 256 - num_special
        for i in range(num_merges):
            # Count pair frequencies
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pair_counts[(word[j], word[j + 1])] += freq

            if not pair_counts:
                break

            # Find best pair
            best_pair = pair_counts.most_common(1)[0][0]
            new_id = 256 + num_special + i

            # Record merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to all words
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, new_id)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

            if verbose and (i + 1) % 1000 == 0:
                pair_bytes = self.vocab[new_id]
                print(f"Merge {i+1}/{num_merges}: {pair_bytes!r} (freq={pair_counts[best_pair]})")

        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items() if isinstance(v, bytes)}
        print(f"Tokenizer trained: {len(self.vocab)} tokens")

    def _apply_merge(
        self, word: Tuple[int, ...], pair: Tuple[int, int], new_id: int
    ) -> Tuple[int, ...]:
        """Apply a single merge operation to a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text: str) -> List[int]:
        """Encode text to token ids using priority-based BPE merging."""
        num_special = len(self.SPECIAL_TOKENS)

        # Build merge priority lookup: pair -> (priority, new_id)
        # Lower priority index = merge first (learned earlier)
        if not hasattr(self, "_merge_priority"):
            self._merge_priority = {}
            for idx, (pair, new_id) in enumerate(self.merges.items()):
                self._merge_priority[pair] = (idx, new_id)

        tokens = []
        words = PRE_TOKENIZE_PATTERN.findall(text)
        for word in words:
            ids = [b + num_special for b in word.encode("utf-8")]
            # Repeatedly merge the highest-priority (lowest index) pair
            while len(ids) >= 2:
                # Find the pair with the lowest merge priority
                best_pair = None
                best_priority = float("inf")
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self._merge_priority:
                        p, _ = self._merge_priority[pair]
                        if p < best_priority:
                            best_priority = p
                            best_pair = pair
                if best_pair is None:
                    break  # No more merges possible
                _, new_id = self._merge_priority[best_pair]
                # Apply this merge everywhere in ids
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i + 1] == best_pair[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            tokens.extend(ids)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to text."""
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                val = self.vocab[token_id]
                if isinstance(val, bytes):
                    byte_chunks.append(val)
                # skip special tokens in output
            else:
                byte_chunks.append(b"\xef\xbf\xbd")  # replacement char
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save tokenizer to directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Save merges as list of pairs
        merges_list = [{"pair": list(k), "new_id": v} for k, v in self.merges.items()]
        with open(p / "merges.json", "w") as f:
            json.dump(merges_list, f)
        # Save vocab (convert bytes keys to hex strings)
        vocab_serializable = {}
        for k, v in self.vocab.items():
            if isinstance(v, bytes):
                vocab_serializable[str(k)] = v.hex()
            else:
                vocab_serializable[str(k)] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        with open(p / "vocab.json", "w") as f:
            json.dump(vocab_serializable, f)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from directory."""
        p = Path(path)
        tok = cls()
        with open(p / "merges.json") as f:
            merges_list = json.load(f)
        tok.merges = {tuple(m["pair"]): m["new_id"] for m in merges_list}
        with open(p / "vocab.json") as f:
            vocab_raw = json.load(f)
        tok.vocab = {}
        for k, v in vocab_raw.items():
            try:
                tok.vocab[int(k)] = bytes.fromhex(v)
            except ValueError:
                tok.vocab[int(k)] = v.encode("utf-8")
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items() if isinstance(v, bytes)}
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
