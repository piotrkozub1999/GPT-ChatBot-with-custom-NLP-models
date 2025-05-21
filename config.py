import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

BLOCK_SIZE = 16  # Length of input and output sequence
BATCH_SIZE = 32
EMBEDDING_DIM = 256
EMBEDDING_DIM_SQAUD = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.2
NUM_EPOCHS = 20
NUM_EPOCHS_LORA = 15
ACCUMULATION_STEPS = 4
MAX_VOCAB_SIZE = 60000
MAX_LENGTH = 50
INPUT_LIMIT = 1024
TEMPERATURE = 0.4
REPETITION_PENALTY = 1.2
SELECTED_MODELS = ["CustomGPT", "PretrainedGPT", "PretrainedGPT_LoRA"]
FINETUNED_GPT2_LARGE_PATH = "./Models/PretrainedGPT2_FineTuned_on_Squad"
FINETUNED_GPT2_LARGE_SAVE_PATH = "./Models/PretrainedGPT2_FineTuned_on_Squad_checkpoints"

SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<|endoftext|>', "<question>", "<context>", "<answer>", "<no_answer>"]
UNK_IDX, PAD_IDX, EOS_IDX, QUESTION_IDX, CONTEXT_IDX, ANSWER_IDX, NO_ANSWER_IDX = 0, 1, 2, 4, 5, 6, 7

dataset_name = "WikiText103"

# Define custom fonts
MESSAGE_FONT = ("Helvetica", 14)
TYPING_BUBBLE_FONT = ("Helvetica", 13, "bold")
BUTTON_FONT = ("Arial", 15)
LABEL_FONT = ("Helvetica", 14)

