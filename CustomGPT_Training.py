import torch
import os
import math
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from collections import Counter
from datasets import load_from_disk
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from CustomGPT import GPT, GPTDataset
from config import (DEVICE, BLOCK_SIZE, BATCH_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
                    NUM_EPOCHS, ACCUMULATION_STEPS, SPECIAL_SYMBOLS, PAD_IDX, MAX_VOCAB_SIZE)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


def load_data(path):
    dataset = load_from_disk(path)
    train_data = [line for line in dataset['train']['text'] if line.strip()]  # Remove empty lines
    val_data = [line for line in dataset['validation']['text'] if line.strip()]
    return train_data, val_data


# Function to tokenize text
def yield_tokens(data_iter, tokenizer):
    for line in data_iter:
        yield tokenizer.encode(line).tokens


# Create WordPiece tokenizer
def build_wordpiece_tokenizer(train_data, vocab_size, special_tokens):
    tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),  # Unicode normalization
        normalizers.Lowercase(),  # Convert to lowercase
        normalizers.StripAccents()  # Remove accents
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Tokenize based on whitespace
    tokenizer.decoder = decoders.WordPiece()

    # Trainer for building the vocabulary
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )

    # Build the vocabulary
    tokenizer.train_from_iterator(train_data, trainer=trainer)
    tokenizer.add_special_tokens(special_tokens)  # Add special tokens

    return tokenizer


# Prune vocabulary and update tokenizer
def prune_vocab_and_update_tokenizer(tokenizer, train_data, min_freq=5):
    """
    Prunes the tokenizer's vocabulary by removing rare tokens based on their frequency.
    """
    # Count tokens
    token_counter = Counter()
    for text in tqdm(train_data, desc="Counting tokens"):
        tokens = tokenizer.encode(text).tokens
        token_counter.update(tokens)

    # Filter tokens that meet the frequency requirement
    valid_tokens = [token for token, count in token_counter.items() if count >= min_freq]

    print("Vocabulary size before pruning:", tokenizer.get_vocab_size())
    print("Number of tokens after pruning:", len(valid_tokens))

    # Retrain the tokenizer with the pruned tokens
    trainer = trainers.WordPieceTrainer(
        vocab_size=len(valid_tokens),
        special_tokens=SPECIAL_SYMBOLS,
    )
    tokenizer.train_from_iterator(valid_tokens, trainer=trainer)

    print("New vocabulary size:", tokenizer.get_vocab_size())
    return tokenizer


def collate_batch(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = [torch.tensor(seq, dtype=torch.long) for seq in src_batch]
    tgt_batch = [torch.tensor(seq, dtype=torch.long) for seq in tgt_batch]
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch.to(DEVICE), tgt_batch.to(DEVICE)


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask


# Function to create masks
def create_mask(src, tgt):
    """
    Creates masks for both source (padding) and target (subsequent) sequences.
    Args:
        src: Source sequence (used to create padding mask).
        tgt: Target sequence (used to create subsequent mask).
    Returns:
        tgt_mask: Mask for the target sequence (ensures no peeking at future tokens).
        src_padding_mask: Padding mask for source sequence (padding tokens are ignored).
        tgt_padding_mask: Padding mask for target sequence.
    """
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)
    # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)
    # Create subsequent mask for target sequence
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)

    # Create padding masks for both source and target sequences
    src_padding_mask = (src == PAD_IDX).to(DEVICE)
    tgt_padding_mask = (tgt == PAD_IDX).to(DEVICE)

    return tgt_mask, src_padding_mask, tgt_padding_mask


def calculate_scheduler_steps(dataset_len, batch_size, accumulation_steps, epochs):
    num_batches_per_epoch = math.ceil(dataset_len / batch_size)
    print("Number of batches: ", num_batches_per_epoch)
    steps_per_epoch = math.ceil(num_batches_per_epoch / accumulation_steps)
    print("Number of steps per epoch: ", steps_per_epoch)
    total_steps = steps_per_epoch * epochs
    print("Total amount of steps: ", total_steps)
    return total_steps


def train_epoch(model, dataloader, criterion, optimizer, scheduler, accumulation_steps=4):
    model.train()
    total_loss = 0

    # Progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training", leave=False)

    # Iterate through the dataloader
    optimizer.zero_grad()
    for batch_idx, (src_batch, tgt_batch) in progress_bar:
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input)

        # Move data to GPU
        src_batch, tgt_input, tgt_output = src_batch.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)

        # Forward pass
        logits = model(src_batch, tgt_input, src_mask=None, tgt_mask=tgt_mask)

        # Calculate loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss = loss / accumulation_steps  # Normalize for gradient accumulation
        loss.backward()

        # Update optimizer after the specified number of steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()

        # Update statistics
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix(batch_loss=loss.item() * accumulation_steps, avg_loss=total_loss / (batch_idx + 1))

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0

    # Progress bar
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation", leave=False)

    with torch.no_grad():
        for batch_idx, (src_batch, tgt_batch) in progress_bar:
            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_batch, tgt_input)

            src_batch, tgt_input, tgt_output = src_batch.to(DEVICE), tgt_input.to(DEVICE), tgt_output.to(DEVICE)

            logits = model(src_batch, tgt_input, src_mask=None, tgt_mask=tgt_mask)

            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

            # Update progress bar description
            progress_bar.set_postfix(batch_loss=loss.item(), avg_loss=total_loss / (batch_idx + 1))

    return total_loss / len(dataloader)


def training(start_epoch, num_epochs, model, train_dataloader, val_dataloader, criterion,
             optimizer, scheduler, ACCUMULATION_STEPS):
    # Simulated batch_size = BATCH_SIZE * ACCUMULATION_STEPS
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, scheduler,
                                 accumulation_steps=ACCUMULATION_STEPS)
        val_loss = evaluate(model, val_dataloader, criterion)

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")


# Converts a tensor of indices to a list of words.
def indices_to_words(indices, tokenizer, pad_token="<pad>"):
    """Converts a tensor of indices to a list of words, preserving <pad> tokens."""
    words = []
    for idx in indices:
        if idx == tokenizer.token_to_id(pad_token):  # Check if it's a <pad> token
            words.append(pad_token)
        else:
            words.append(tokenizer.id_to_token(idx))  # Convert index to token
    return words


def show_batches(num_samples, tokenizer, train_dataloader, val_dataloader):
    # Test the DataLoader
    for src_batch, tgt_batch in train_dataloader:
        print("Train - Source batch shape:", src_batch.shape)
        print("Train - Target batch shape:", tgt_batch.shape)
        break

    for src_batch, tgt_batch in val_dataloader:
        print("Validation - Source batch shape:", src_batch.shape)
        print("Validation - Target batch shape:", tgt_batch.shape)
        break

    for i in range(min(num_samples, src_batch.size(0))):  # Limit to batch size
        src_sample = src_batch[i].cpu()  # Move tensor to CPU
        tgt_sample = tgt_batch[i].cpu()

        # Convert indices to words
        src_words = indices_to_words(src_sample, tokenizer)
        tgt_words = indices_to_words(tgt_sample, tokenizer)

        # Display sequences
        print(f"\nExample {i + 1}:")
        print("Source (src):", src_words)
        print("Target (tgt):", tgt_words)


# Function to save the model
def save_model(model, name, epochs):
    filepath = 'Models\gpt_model_' + name + '_' + str(epochs) + 'Epochs.pth'
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath


# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, epochs, name):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epochs
    }
    path = 'Models\gpt_model_' + name + '_' + str(epochs) + 'Epochs_checkpoint.pth'
    torch.save(checkpoint, path)
    print(f"Model saved to: {path}")
    return path


def show_first_n_tokens(tokenizer: Tokenizer, n: int):
    # Displays the first n tokens in the tokenizer's vocabulary.
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    for token, idx in sorted_vocab[:n]:
        print(f"{idx}: {token}")


def save_vocab(tokenizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tokenizer.save(path)
    print(f"Tokenizer saved to {path}")


def load_vocab(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer file not found: {path}")

    tokenizer = Tokenizer.from_file(path)
    print(f"Tokenizer loaded from {path}")
    return tokenizer


if __name__=='__main__':
    dataset_name = "WikiText103"
    train_data, val_data = load_data("./Datasets/wikitext-103")

    ''' Uncomment when you train modern first time'''
    # # Build WordPiece tokenizer
    # tokenizer = build_wordpiece_tokenizer(train_data, MAX_VOCAB_SIZE, SPECIAL_SYMBOLS)
    #
    # # Prune vocabulary and update tokenizer
    # tokenizer = prune_vocab_and_update_tokenizer(tokenizer, train_data, min_freq=5)
    # save_vocab(tokenizer, "./tokenizer/Tokenizer_WikiText103_60k.json")

    # Load saved vocabulary
    loaded_tokenizer = load_vocab("./tokenizer/Tokenizer_WikiText103_60k.json")
    VOCAB_SIZE = loaded_tokenizer.get_vocab_size()
    print(f"Vocabulary size: ", VOCAB_SIZE)

    # Create Datasets and DataLoaders
    train_dataset = GPTDataset(train_data, loaded_tokenizer, BLOCK_SIZE)
    val_dataset = GPTDataset(val_data, loaded_tokenizer, BLOCK_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Display example sequences
    show_batches(2, loaded_tokenizer, train_dataloader, val_dataloader)

    DATASET_LEN = len(train_dataloader.dataset)
    model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)

    # Calculate total steps
    total_steps = calculate_scheduler_steps(DATASET_LEN, BATCH_SIZE, ACCUMULATION_STEPS, NUM_EPOCHS)

    # Optimizer, loss function, and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=total_steps
    )

    training(0, NUM_EPOCHS, model, train_dataloader, val_dataloader, criterion,
             optimizer, scheduler, ACCUMULATION_STEPS)
    save_model(model, dataset_name, NUM_EPOCHS)
    save_checkpoint(model, optimizer, scheduler, NUM_EPOCHS, dataset_name)