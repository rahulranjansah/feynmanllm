# %% [markdown]
# ## Load the sequence

# %%
# !pip install -q minbpe

# %%
with open("../data/feynman_combined_text.txt", "r") as f:
    text_sequence = f.read()

len(text_sequence)

# %% [markdown]
# ## BPE algorithm

# %% [markdown]
# I am using the [minBPE](https://github.com/karpathy/minbpe) repository to tokenize the sequence of text.

# %%
import sys
sys.path.append('..')

# %% [markdown]
# Start by training the tokenizer on the text sequence that you saved in the previous notebook.

# %%
from minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train(text_sequence, vocab_size=1024)

# %% [markdown]
# Visualize the vocabulary.

# %%
vocab = tokenizer.vocab
vocab

# %% [markdown]
# Test the tokenizer.

# %%
tokenizer.encode("action-reaction experiment")

# %%
tokenizer.decode([345, 367, 45, 275, 345, 375, 1023, 343])

# %% [markdown]
# Add special tokens to the vocabulary. These tokens are going to be used a lot in the fine-tuning step.

# %%
max_vocab_id = list(tokenizer.vocab.keys())[-1]
tokenizer.special_tokens = {
    "<|startoftext|>": max_vocab_id + 1,
    "<|separator|>": max_vocab_id + 2,
    "<|endoftext|>": max_vocab_id + 3,
    "<|unk|>": max_vocab_id + 4
}

# %% [markdown]
# I have more than 618K tokens for training and validation. This is pretty good, but if you can add more, that would be even better.

# %%
# len(tokenizer.encode(text_sequence))

# %%
# Sample first 100k characters to estimate total tokens
sample_size = 1000
sample = text_sequence[:sample_size]
tokens_per_char = len(tokenizer.encode(sample)) / len(sample)
estimated_total = int(tokens_per_char * len(text_sequence))
print(f"Estimated total tokens: {estimated_total}")

# %%
tokenizer.save("../output/tokenizer/my_tokenizer")

# %% [markdown]
# Save the tokenizer

# %%
# import pickle
# from pathlib import Path

# # Create the directory structure
# save_dir = Path("../output/tokenizer")
# save_dir.mkdir(parents=True, exist_ok=True)

# # Save tokenizer
# with open(save_dir / "my_tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)


