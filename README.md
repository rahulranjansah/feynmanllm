# FeynmanLLM

FeynmanLLM is a project designed to train and fine-tune a language model inspired by the works of Richard Feynman. The project processes text data from Feynman's lectures and other sources to create a custom tokenizer, train a GPT-based language model, and fine-tune it for specific tasks.

---

## Features

- **Custom Tokenizer**: A tokenizer is trained using the `minbpe` library to handle the unique vocabulary of Feynman's texts.
- **GPT-Based Model**: A transformer-based GPT model is implemented and trained from scratch.
- **Fine-Tuning**: The model can be fine-tuned on specific datasets for downstream tasks.
- **Checkpointing**: Training progress is saved using checkpoints to avoid retraining from scratch.
- **Data Cleaning**: Includes preprocessing steps to clean and structure raw text data.

