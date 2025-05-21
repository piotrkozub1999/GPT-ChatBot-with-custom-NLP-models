import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from CustomGPT import GPT, GPTDataset
from CustomGPT_Training import (load_data, collate_batch, calculate_scheduler_steps,
                                training, save_model, save_checkpoint, load_vocab)
from CustomGPT_Continue_Learning import load_checkpoint
from config import (DEVICE, BLOCK_SIZE, BATCH_SIZE, EMBEDDING_DIM, NUM_HEADS,
                    NUM_LAYERS, DROPOUT, ACCUMULATION_STEPS, PAD_IDX)

MODEL_CHECKPOINT_SAVE_PATH = "./Models/gpt_model_WikiText103_Squad_25Epochs_checkpoint.pth"
# Number of epochs for further learning
NEW_EPOCHS = 5


def preprocess_data(data, tokenizer, max_length=200):
    processed_data = []

    for sample in data:
        context = sample["context"]
        question = sample["question"]
        answer = sample["answers"]["text"][0] if sample["answers"]["text"] else "<no_answer>"

        # Tokenization with padding and truncation
        source_tokens = tokenizer.encode(f"<question> {question} <context> {context}").ids[:max_length]
        target_tokens = tokenizer.encode(f"<answer> {answer}").ids[:max_length]

        processed_data.append((source_tokens, target_tokens))

    return processed_data


def collate_squad(batch):
    src_batch, tgt_batch = zip(*batch)

    # Find the maximum length in the batch
    max_src_len = max(len(seq) for seq in src_batch)
    max_tgt_len = max(len(seq) for seq in tgt_batch)

    # Pad to the maximum length in the batch
    src_batch = [torch.tensor(seq + [PAD_IDX] * (max_src_len - len(seq)), dtype=torch.long) for seq in src_batch]
    tgt_batch = [torch.tensor(seq + [PAD_IDX] * (max_tgt_len - len(seq)), dtype=torch.long) for seq in tgt_batch]

    src_batch = torch.stack(src_batch).to(DEVICE)
    tgt_batch = torch.stack(tgt_batch).to(DEVICE)

    return src_batch, tgt_batch


def show_samples(dataloader, tokenizer, num_samples=5):
    #Displays sample inputs (source) and labels (target) from the DataLoader
    for i, (src_batch, tgt_batch) in enumerate(dataloader):
        if i >= num_samples:  # Limit to the number of samples
            break

        print(f"\nExample {i + 1}:")

        # Convert tokens to text
        src_text = tokenizer.decode(src_batch[0].tolist(), skip_special_tokens=False)
        tgt_text = tokenizer.decode(tgt_batch[0].tolist(), skip_special_tokens=False)

        print("Input (source):", src_text)
        print("Labels (target):", tgt_text)
        print("-" * 50)


if __name__ == '__main__':

    train_data, val_data = load_data("Datasets\wikitext-103")

    # Load vocabulary
    tokenizer = load_vocab("./tokenizer/Tokenizer_WikiText103_60k.json")
    VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary size: ", VOCAB_SIZE)

    # Create Datasets and DataLoaders
    train_dataset = GPTDataset(train_data, tokenizer, BLOCK_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    DATASET_LEN = len(train_dataloader.dataset)

    ##################  INITIALIZING MODEL AND LEARNING PARAMETERS  ##################
    # Initializing model
    model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)

    # Setting criterion, optimizer and temp scheduler before loading from checkpoint
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tmp_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=1)  # Placeholder

    # Loading state of model, optimizer and scheduler
    start_epoch = load_checkpoint(model, optimizer, tmp_scheduler, path=MODEL_CHECKPOINT_SAVE_PATH, device=DEVICE)

    total_steps_old = calculate_scheduler_steps(DATASET_LEN, BATCH_SIZE, ACCUMULATION_STEPS, start_epoch)


    ##################  UPDATING DICTIONARY  ##################
    # squad = load_dataset("squad_v2")
    squad = load_from_disk("Datasets\squad_v2")

    # Train and validation data processing
    train_data = preprocess_data(squad["train"], tokenizer)
    val_data = preprocess_data(squad["validation"], tokenizer)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_squad)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_squad)

    # Displaying example data
    print("Training data example:")
    show_samples(train_dataloader, tokenizer, num_samples=6)

    DATASET_LEN = len(train_dataloader.dataset)

    #Setting new epochs number
    num_epochs = start_epoch + NEW_EPOCHS
    total_steps = total_steps_old + calculate_scheduler_steps(DATASET_LEN, BATCH_SIZE, ACCUMULATION_STEPS, NEW_EPOCHS)

    # Creating new scheduler with new steps_num and moving previous state from older one
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)
    scheduler.last_epoch = tmp_scheduler.last_epoch  # Keep old schedulers steps number
    scheduler._schedule_phases = tmp_scheduler._schedule_phases  # Copy scheduler phases


    ##################  TRAINING AND SAVING MODEL  ##################
    training(start_epoch, num_epochs, model, train_dataloader, val_dataloader, criterion,
             optimizer, scheduler, ACCUMULATION_STEPS)

    dataset_name = "WikiText103_Squad"
    save_model(model, dataset_name, num_epochs)
    save_checkpoint(model, optimizer, scheduler, num_epochs, dataset_name)

