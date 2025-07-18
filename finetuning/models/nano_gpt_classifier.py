import torch
import torch.nn as nn
from pretraining.scripts.model import GPT

class NanoGPTClassifier(nn.Module):
    def __init__(self, gpt_config, num_classes=2):
        super().__init__()
        self.gpt = GPT(gpt_config)
        self.classifier = nn.Linear(gpt_config.n_embd, num_classes)

    def forward(self, x, labels=None):
        # x: (batch, seq_len)
        logits, _ = self.gpt(x)
        # Use the last token's hidden state for classification
        # Get hidden states from the last transformer block
        with torch.no_grad():
            # Forward through embedding and transformer blocks
            tok_emb = self.gpt.transformer.wte(x)
            pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.gpt.transformer.wpe(pos)
            h = self.gpt.transformer.drop(tok_emb + pos_emb)
            for block in self.gpt.transformer.h:
                h = block(h)
            h = self.gpt.transformer.ln_f(h)
        last_hidden = h[:, -1, :]  # (batch, n_embd)
        cls_logits = self.classifier(last_hidden)  # (batch, num_classes)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(cls_logits, labels)
        return cls_logits, loss

    def load_gpt_weights(self, state_dict):
        self.gpt.load_state_dict(state_dict) 