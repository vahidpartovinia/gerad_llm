import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Classifier(nn.Module):
    def __init__(self, model_dir, num_classes=2):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.num_classes = num_classes
        # For SST-2, class tokens are '0' and '1'
        self.class_token_ids = [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(num_classes)]

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids)
        last_token_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
        class_logits = torch.stack([
            last_token_logits[:, class_id] for class_id in self.class_token_ids
        ], dim=1)  # (batch, num_classes)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(class_logits, labels)
        return class_logits, loss

    @property
    def device(self):
        return next(self.parameters()).device

    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)


def gpt2_collate_fn(batch, tokenizer, max_length):
    texts, labels = zip(*batch)
    encodings = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    input_ids = encodings["input_ids"]
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, labels 