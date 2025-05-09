"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from tqdm import tqdm
from .base import Tokenizer, get_stats, merge
import json


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size, verbose=False):
        """Train the tokenizer on the given text.

        Args:
            text (str): The text to train on
            vocab_size (int): The desired vocabulary size (must be >= 256)
            verbose (bool): Whether to print progress information
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        if not text.strip():
            raise ValueError("Cannot train on empty text")

        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        if len(text_bytes) < 2:
            raise ValueError("Text is too short for training (need at least 2 bytes)")

        ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {}  # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes

        for i in tqdm(range(num_merges), total=num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids)
            if not stats:
                if verbose:
                    print(f"No more pairs to merge after {i} merges")
                break

            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = merge(ids, pair, idx)
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab   # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def save(self, filepath):
        """Save tokenizer state to a JSON file"""
        # Convert bytes objects to lists for JSON serialization
        vocab_serializable = {k: list(v) for k, v in self.vocab.items()}
        # Convert tuple keys to strings for JSON serialization
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        state = {
            'merges': merges_serializable,
            'vocab': vocab_serializable
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load(self, filepath):
        """Load tokenizer state from a JSON file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        # Convert string keys back to tuples
        self.merges = {tuple(map(int, k.split(','))): v
                      for k, v in state['merges'].items()}
        self.vocab = {int(k): bytes(v) for k, v in state['vocab'].items()}
