import os
import torch
import time
import matplotlib.pyplot as plt
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling, TrainerCallback, BitsAndBytesConfig, AdamW)
from config import FINETUNED_GPT2_LARGE_PATH,FINETUNED_GPT2_LARGE_SAVE_PATH, EMBEDDING_DIM_SQAUD, NUM_EPOCHS_LORA

import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION
MODEL_NAME = "gpt2-large"
DATASET_PATH = "./Datasets/squad_v2"
# NUM_EPOCHS = 15

# 6. CALLBACK FOR TRACKING LOSS & TIME
class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_steps=1000):
        self.train_losses = []
        self.epoch_start_time = None
        self.log_steps = log_steps  # Loss log steps frequency

    def on_step_end(self, args, state, control, **kwargs):
        """Log loss after log_steps reached"""
        if state.global_step % self.log_steps == 0 and state.log_history:
            last_log = state.log_history[-1]
            train_loss = last_log.get("loss")
            if train_loss is not None:
                self.train_losses.append((state.global_step, train_loss))

    def save_loss_plot(self):
        """Lose plot after log_steps and every epoch."""
        plt.figure(figsize=(8, 5))
        # Train Loss
        if self.train_losses:
            steps, train_loss_vals = zip(*self.train_losses)
            plt.plot(steps, train_loss_vals, label='Train Loss (per step)')

        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.grid()
        plt.savefig("./Models/loss_plot_3.png")
        print(f"📊 Loss plot saved at loss_plot.png")




# 1. LOADING THE DATASET
dataset = load_from_disk(DATASET_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no native padding token

#### First Learning
# model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, device_map="auto")
#### Further learning
model = GPT2LMHeadModel.from_pretrained(FINETUNED_GPT2_LARGE_PATH, device_map="auto")

# 4. LORA: PARAMETER-EFFICIENT FINE-TUNING
peft_config = LoraConfig(
    r=16,  # LoRA hidden dimension
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"  # GPT-2 is an autoregressive model
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Checking how many parameters are trainable

def preprocess_data(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""

    prompt = f"Question: {question}\nAnswer: {answer}"
    max_context_length = 200 - len(tokenizer(prompt)["input_ids"]) - 1  # Setting max context length by subtraction prompt length and 1 for spacing

    truncated_context = tokenizer.decode(tokenizer(context, truncation=True, max_length=max_context_length)["input_ids"], skip_special_tokens=True)
    input_text = f"Context: {truncated_context}\n{prompt}"

    encoding = tokenizer(input_text, truncation=True, padding="max_length", max_length=EMBEDDING_DIM_SQAUD)

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": encoding["input_ids"],
    }

# Tokenizing the dataset for each split separately
tokenized_dataset = {
    split: dataset[split].map(preprocess_data, remove_columns=["id", "title", "context", "question", "answers"])
    for split in ["train", "validation"]
}

# Data collator (token masking)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# FUNCTION TO DISPLAY EXAMPLES
def show_sample_batches(dataset, num_samples=3):
    print("\n🔍 Example Batches:")
    for i in range(num_samples):
        index = 100*i
        if index<len(dataset):
            example = dataset[index]
            decoded_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            print(f"\n📌 Sample {index+1}:\n{decoded_text}")


loss_logger = LossLoggerCallback()

# 7. TRAINING SETTINGS
training_args = TrainingArguments(
    output_dir=FINETUNED_GPT2_LARGE_SAVE_PATH,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # Reduced batch size for 8GB VRAM
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Larger effective batch size
    evaluation_strategy="epoch",
    save_strategy="epoch", # Checkpoint after every epoch
    logging_strategy="steps",
    logging_steps=1,
    save_total_limit=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    max_grad_norm=1,
    num_train_epochs=NUM_EPOCHS_LORA,
    warmup_steps=500,
    fp16=True,  # Using bfloat16 instead of fp16 - demands newer version of torch
    logging_dir="./logs",
    report_to="none",
    load_best_model_at_end=True,
)

# 8. INITIALIZING THE TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[loss_logger],# Adding a callback to track time
)



# 9. MODEL TRAINING
# Show a few batches before training
show_sample_batches(tokenized_dataset["train"], num_samples=20)

trainer.train()

# 10. SAVING THE MODEL AFTER TRAINING
model.save_pretrained(FINETUNED_GPT2_LARGE_SAVE_PATH)
torch.save(model.state_dict(), './Models/PretrainedGPT2_FineTuned_on_Squad.pt')
print(f"Model saved at {FINETUNED_GPT2_LARGE_SAVE_PATH}")
loss_logger.save_loss_plot()



