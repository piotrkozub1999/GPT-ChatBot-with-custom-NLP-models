import torch
from CustomGPT_Training import generate_square_subsequent_mask
from CustomGPT_Training import load_vocab
from CustomGPT import GPT
from torch.nn.functional import softmax
from config import (DEVICE, BLOCK_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
                    MAX_LENGTH, TEMPERATURE, REPETITION_PENALTY)


def penalize_repetition(logits, generated_tokens, penalty=1.2):
    for token in set(generated_tokens):
        logits[0, token] /= penalty
    return logits

def customGPT_generate_text(model, tokenizer, device, prompt, max_length=MAX_LENGTH,
                               temperature=TEMPERATURE, repetition_penalty=REPETITION_PENALTY):
    """
    Generates text using the model while penalizing token repetitions.
    """
    model.eval()

    # Tokenize the initial prompt
    tokenized_output = tokenizer.encode(prompt)
    token_indices = torch.tensor([tokenized_output.ids], dtype=torch.long, device=device)

    generated_indices = token_indices.tolist()[0]  # Stores generated tokens

    for _ in range(max_length):
        # Create a mask for the current sequence
        seq_len = len(generated_indices)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

        # Predict the next token
        with torch.no_grad():
            logits = model(token_indices, token_indices, tgt_mask=tgt_mask)

        # Get logits for the last token
        next_token_logits = logits[0, -1, :] / temperature

        # Penalize repetitions
        for token in set(generated_indices):
            next_token_logits[token] /= repetition_penalty

        # Choose the next token
        next_token_id = torch.multinomial(softmax(next_token_logits, dim=-1), num_samples=1).item()

        # Check if the end-of-text token is generated
        if next_token_id == tokenizer.token_to_id('<|endoftext|>'):
            break

        # Add the token to the sequence
        generated_indices.append(next_token_id)

        # Update token_indices
        token_indices = torch.tensor([generated_indices], dtype=torch.long, device=device)

    # Decode tokens into text
    generated_text = tokenizer.decode(generated_indices)

    generated_text = generated_text[len(prompt):].strip()
    return generated_text


if __name__=='__main__':

    loaded_tokenizer = load_vocab("./tokenizer/Tokenizer_WikiText103_60k.json")
    VOCAB_SIZE = loaded_tokenizer.get_vocab_size()
    print(f"Number of SQUAD tokens: : {VOCAB_SIZE}")

    MODEL_SAVE_PATH = "./Models/gpt_model_WikiText103_Squad_25Epochs.pth"

    model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(DEVICE)
    model.eval()

    # Seed text (prompt)
    prompt = "What type of music Beyonce sing?"

    # Call the text generation function
    generated_text = customGPT_generate_text(model, loaded_tokenizer, device=DEVICE, prompt=prompt,
                                                max_length=MAX_LENGTH, temperature=TEMPERATURE)
    print("Seed Text:", prompt)
    print("Generated Text:", generated_text)