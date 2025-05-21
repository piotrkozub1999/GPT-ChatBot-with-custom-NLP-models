import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomGPT import GPT, GPTDataset
from CustomGPT_Training import load_data, load_vocab, collate_batch, calculate_scheduler_steps, training
from config import (DEVICE, BLOCK_SIZE, BATCH_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, NUM_EPOCHS,
                    ACCUMULATION_STEPS, PAD_IDX)


MODEL_CHECKPOINT_SAVE_PATH = "./Models/gpt_model_WikiText103_20Epochs_checkpoint.pth"
# Number of epochs for further learning
NEW_EPOCHS = 3

def load_checkpoint(model, optimizer, scheduler, path, device=DEVICE):
    checkpoint = torch.load(path, map_location=device)

    # Loading model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Loading optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Loading scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Loading epochs already trained
    epoch = checkpoint.get('epoch', 0)

    print(f"Model loaded from: {path} (Epoch: {epoch})")
    return epoch


if __name__ == '__main__':
    dataset_name = "WikiText103"
    train_data, val_data = load_data("./Datasets/wikitext-103")

    # Load saved vocabulary
    loaded_tokenizer = load_vocab("./tokenizer/Tokenizer_WikiText103_60k.json")
    VOCAB_SIZE = loaded_tokenizer.get_vocab_size()
    print(f"Vocabulary size: ", VOCAB_SIZE)

    # Create Datasets and DataLoaders
    train_dataset = GPTDataset(train_data, loaded_tokenizer, BLOCK_SIZE)
    val_dataset = GPTDataset(val_data, loaded_tokenizer, BLOCK_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    DATASET_LEN = len(train_dataloader.dataset)

    # Initializing model
    model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)

    # Setting criterion, optimizer and temp scheduler before loading from checkpoint
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tmp_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=1)  # Placeholder

    # Loading state of model, optimizer and scheduler
    start_epoch = load_checkpoint(model, optimizer, tmp_scheduler, path=MODEL_CHECKPOINT_SAVE_PATH, device=DEVICE)

    # Setting new epochs number
    num_epochs = start_epoch + NEW_EPOCHS
    total_steps = calculate_scheduler_steps(DATASET_LEN, BATCH_SIZE, ACCUMULATION_STEPS, num_epochs)

    # Creating new scheduler with new steps_num and moving previous state from older one
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)
    scheduler.last_epoch = tmp_scheduler.last_epoch  # Keep old schedulers steps number
    scheduler._schedule_phases = tmp_scheduler._schedule_phases  # Copy scheduler phases

    training(start_epoch, num_epochs, model, train_dataloader, val_dataloader, criterion,
             optimizer, scheduler, ACCUMULATION_STEPS)