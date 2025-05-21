import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import PAD_IDX


def generate_positional_encoding(block_size, embedding_dim, device):
    position = torch.arange(0, block_size, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    pe = torch.zeros(block_size, embedding_dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Add batch dimension


# Function to generate samples (source and target)
def get_sample(block_size, tokens):
    # Returns a pair of sequences (source, target) for the language model
    sample_len = len(tokens)
    random_sample_stop = sample_len - block_size

    if random_sample_stop >= 1:
        start_idx = torch.randint(0, random_sample_stop, (1,)).item()
        stop_idx = start_idx + block_size
        src_sequence = tokens[start_idx:stop_idx]
        tgt_sequence = tokens[start_idx + 1:stop_idx + 1]
    else:
        # If the sample is shorter than block_size, use the entire sequence
        src_sequence = tokens[:block_size]
        tgt_sequence = tokens[1:block_size] + ['<|endoftext|>']

    return src_sequence, tgt_sequence



class GPTDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the text
        tokens = self.tokenizer.encode(self.data[idx]).ids[:self.block_size]  #  Limit to block size
        src_sequence = tokens[:-1]  # Source sequence
        tgt_sequence = tokens[1:]   # Target sequence
        return torch.tensor(src_sequence, dtype=torch.long), torch.tensor(tgt_sequence, dtype=torch.long)


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.positional_encoding = nn.Parameter(torch.zeros(1, block_size, embedding_dim))
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights after creating all layers
        self.init_weights()

    def init_weights(self):
        # Initialize weights for model layers
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Dynamic positions
        positional_encoding = generate_positional_encoding(src.size(1), self.embedding.embedding_dim, src.device)
        src_emb = self.embedding(src) + positional_encoding
        tgt_emb = self.embedding(tgt) + positional_encoding[:, :tgt.size(1), :]

        src_emb = self.norm(src_emb)
        tgt_emb = self.norm(tgt_emb)

        transformer_output = self.transformer(
            src_emb.transpose(0, 1), tgt_emb.transpose(0, 1), src_mask, tgt_mask, memory_mask
        )

        logits = self.fc_out(transformer_output.transpose(0, 1))
        return logits