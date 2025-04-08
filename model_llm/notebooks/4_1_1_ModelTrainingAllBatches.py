# %% [markdown]
# ## Load the tokenizer

# %%
# import sys
# sys.path.append('..')

# %%
from model_llm.minbpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.load("model_llm/output/tokenizer/my_tokenizer.model")


def get_vocab_size(tokenizer: BasicTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)

# %% [markdown]
# ## Create the model

# %%
import torch
torch.manual_seed(3647)

# %%
from model_llm.transformer.model import GPTLanguageModel

block_size = 256
n_embd = 512
n_head = 12
n_layer = 4
dropout = 0.2
batch_size = 64
vocab_size = get_vocab_size(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device
).to(device)
model = torch.compile(model)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# %% [markdown]
# ## Data preparation

# %% [markdown]
# ### 1. Load the data

# %%
with open("model_llm/data/feynman_combined_text.txt", "r") as f:
    text_sequence = f.read()

encoded_text_sequence = tokenizer.encode(text_sequence)
# len(encoded_text_sequence)

# %% [markdown]
# ### 2. Split it into train and test

# %%
data = torch.tensor(encoded_text_sequence, dtype=torch.long)
split_index = int(0.9*len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# %% [markdown]
# ### 3. Data loader

# %%
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[index:index + self.block_size]
        y = self.data[index + 1:index + self.block_size + 1]
        return x, y


def get_dataloaders(
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = TextDataset(train_data.to(device), block_size)
    val_dataset = TextDataset(val_data.to(device), block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

# %%
train_loader, val_loader = get_dataloaders(
    train_data=train_data,
    val_data=val_data,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)
x, y = next(iter(train_loader))
x.shape, y.shape

# %% [markdown]
# ### 4. Training

# %%
from typing import Dict


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int
) -> Dict[str, float]:
    output = {}
    model.eval()

    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            with torch.no_grad():
                _, loss = model(x, y)
            losses[i] = loss.item()
        output[split] = losses.mean().item()

    model.train()
    return output

# %% [markdown]
# Training modularized to save and load checkpoints

# %%
import torch
from tqdm import tqdm
import os

def save_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    loss: float,
    train_losses: list,
    val_losses: list,
    file_path: str = "checkpoint.pth"
) -> None:
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
    torch.save(checkpoint, file_path)

# %%
def load_checkpoint(file_path, model, optimizer):
    if not os.path.exists(file_path):
        return 0, 0, [], []  # Start fresh if no checkpoint exists

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch_idx = checkpoint['batch_idx'] + 1  # Resume from next batch
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])

    return start_epoch, start_batch_idx, train_losses, val_losses

# %%
def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    max_iters=1,
    eval_interval=100,
    eval_iters=200,
    checkpoint_path="checkpoint.pth",
    checkpoint_interval=500,  # Save checkpoint every N batches
    target_train_loss=1.5  # Early stopping threshold
):
    # Try to load existing checkpoint
    start_epoch, start_batch_idx, train_losses, val_losses = load_checkpoint(
        checkpoint_path, model, optimizer
    )

    for iteration in range(start_epoch, max_iters):
        # Skip already processed batches in the current epoch
        train_loader_iter = enumerate(train_loader)
        if iteration == start_epoch and start_batch_idx > 0:
            for _ in range(start_batch_idx):
                next(train_loader_iter)

        for batch_idx, (x_batch, y_batch) in tqdm(train_loader_iter, desc=f"Epoch {iteration + 1}"):
            # Evaluation
            if (batch_idx % eval_interval == 0) or (batch_idx == len(train_loader) - 1):
                losses = estimate_loss(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iters=min(eval_iters, len(val_loader))
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                print(
                    f"iteration {iteration} / step {batch_idx}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

                # Early stopping condition
                if losses['train'] < target_train_loss:
                    print(f"Early stopping triggered: Train loss {losses['train']:.4f} < {target_train_loss}")
                    # Save the final checkpoint before exiting
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=iteration,
                        batch_idx=batch_idx,
                        loss=losses['train'],
                        train_losses=train_losses,
                        val_losses=val_losses,
                        file_path=checkpoint_path
                    )
                    return  # Exit the training loop

            # Training step
            logits, loss = model(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Save checkpoint periodically
            if batch_idx % checkpoint_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=iteration,
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    file_path=checkpoint_path
                )

        # Save checkpoint at end of epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=iteration,
            batch_idx=len(train_loader) - 1,  # Mark epoch as complete
            loss=loss.item(),
            train_losses=train_losses,
            val_losses=val_losses,
            file_path=checkpoint_path
        )

# %%
def train_model(
    model,
    optimizer,
    train_loader,
    val_loader,
    max_iters=1,
    eval_interval=100,
    eval_iters=200,
    checkpoint_path="checkpoint.pth",
    checkpoint_interval=500  # Save checkpoint every N batches
    train_test_loss=1.5
):
    # Try to load existing checkpoint
    start_epoch, start_batch_idx, train_losses, val_losses = load_checkpoint(
        checkpoint_path, model, optimizer
    )

    for iteration in range(start_epoch, max_iters):
        # Skip already processed batches in the current epoch
        train_loader_iter = enumerate(train_loader)
        if iteration == start_epoch and start_batch_idx > 0:
            for _ in range(start_batch_idx):
                next(train_loader_iter)

        for batch_idx, (x_batch, y_batch) in (tqdm(train_loader_iter, desc=f"Epoch {iteration + 1}")):
            # Evaluation
            if (batch_idx % eval_interval == 0) or (batch_idx == len(train_loader) - 1):
                losses = estimate_loss(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iters=min(eval_iters, len(val_loader))
                )
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                print(
                    f"iteration {iteration} / step {batch_idx}: "
                    f"train loss {losses['train']:.4f}, "
                    f"val loss {losses['val']:.4f}"
                )

            # Training step
            logits, loss = model(x_batch, y_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Save checkpoint periodically
            if batch_idx % checkpoint_interval == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=iteration,
                    batch_idx=batch_idx,
                    loss=loss.item(),
                    train_losses=train_losses,
                    val_losses=val_losses,
                    file_path=checkpoint_path
                )

        # Save checkpoint at end of epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=iteration,
            batch_idx=len(train_loader) - 1,  # Mark epoch as complete
            loss=loss.item(),
            train_losses=train_losses,
            val_losses=val_losses,
            file_path=checkpoint_path
        )

# %%
# Train with vals
max_iters = 1  # Total epochs you want to run
eval_interval = 100
eval_iters = 200
learning_rate = 3e-4
checkpoint_path = "model_llm/output/pre_training/run_4/checkpoint.pth"

os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
train_loader, val_loader = get_dataloaders(
    train_data=train_data,
    val_data=val_data,
    block_size=block_size,
    batch_size=batch_size,
    device=device
)

train_model(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    max_iters=max_iters,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
    checkpoint_path=checkpoint_path
)

# %% [markdown]
# done
#

# %%


# %%
# def save_checkpoint(
#     model: GPTLanguageModel,
#     optimizer: torch.optim.Optimizer,
#     epoch: int,
#     loss: float,
#     file_path: str = "checkpoint.pth"
# ) -> None:
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
#     torch.save(checkpoint, file_path)

# %%


# %%
# # loading the checkpoint where stopped
# def load_checkpoint(file_path, model, optimizer):
#     checkpoint = torch.load(file_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
#     return start_epoch

# %%
# max_iters = 1 # epoch
# eval_interval = 100
# eval_iters = 200
# learning_rate = 3e-4

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# train_loader, val_loader = get_dataloaders(
#     train_data=train_data,
#     val_data=val_data,
#     block_size=block_size,
#     batch_size=batch_size,
#     device=device
# )

# train_losses = []
# val_losses = []

# for iteration in range(max_iters):
#     for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {iteration + 1}")):
#         # Evaluation
#         if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) - 1:
#             losses = estimate_loss(
#                 model=model,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 eval_iters=min(eval_iters, len(val_loader))
#             )
#             train_losses.append(losses['train'])
#             val_losses.append(losses['val'])

#             print(
#                 f"iteration {iteration} / step {batch_idx}: "
#                 f"train loss {losses['train']:.4f}, "
#                 f"val loss {losses['val']:.4f}"
#             )

#         # Training step
#         logits, loss = model(x_batch, y_batch)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#     # Save checkpoint
#     save_checkpoint(
#         model=model,
#         optimizer=optimizer,
#         epoch=iteration,
#         loss=loss.item(),
#         file_path=f"model_llm/output/pre_training/run_4/checkpoint_{iteration}.pth"
#     )

# %%


# %%
# max_iters = 1 # epoch
# eval_interval = 100
# eval_iters = 200
# learning_rate = 3e-4

# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# train_loader, val_loader = get_dataloaders(
#     train_data=train_data,
#     val_data=val_data,
#     block_size=block_size,
#     batch_size=batch_size,
#     device=device
# )

# train_losses = []
# val_losses = []

# for iteration in range(max_iters):
#     for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {iteration + 1}")):
#         # Evaluation
#         if batch_idx % eval_interval == 0 or batch_idx == len(train_loader) - 1:
#             losses = estimate_loss(
#                 model=model,
#                 train_loader=train_loader,
#                 val_loader=val_loader,
#                 eval_iters=min(eval_iters, len(val_loader))
#             )
#             train_losses.append(losses['train'])
#             val_losses.append(losses['val'])

#             print(
#                 f"iteration {iteration} / step {batch_idx}: "
#                 f"train loss {losses['train']:.4f}, "
#                 f"val loss {losses['val']:.4f}"
#             )

#         # Training step
#         logits, loss = model(x_batch, y_batch)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#     # Save checkpoint
#     save_checkpoint(
#         model=model,
#         optimizer=optimizer,
#         epoch=iteration,
#         loss=loss.item(),
#         file_path=f"model_llm/output/pre_training/run_4/checkpoint_{iteration}.pth"
#     )

# %%


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='o')
plt.xlabel("Evaluation Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid()
plt.show()

# %%
import matplotlib.pyplot as plt

# Ensure train_losses and val_losses are not empty
if len(train_losses) == 0 or len(val_losses) == 0:
    print("Error: train_losses or val_losses is empty. Ensure they are being updated during training.")
else:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker='o', linestyle='-', color='blue')
    plt.plot(val_losses, label="Validation Loss", marker='o', linestyle='--', color='orange')
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
input_tokens = tokenizer.encode("Potential Energy ")
input_tokens = torch.tensor(
    input_tokens, dtype=torch.long).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model.generate(input_tokens=input_tokens, max_new_tokens=100)

print(tokenizer.decode(output[0].tolist()))


