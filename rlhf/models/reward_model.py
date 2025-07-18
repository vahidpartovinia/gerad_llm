import torch
import torch.nn as nn
import transformers

class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=128):
        super().__init__()
        self.base_model = base_model  # e.g., NanoGPT or GPT-2
        self.scorer = nn.Sequential(
            nn.Linear(self.base_model.config.n_embd, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    import transformers

    def forward(self, input_ids, attention_mask=None):
        if isinstance(self.base_model, transformers.PreTrainedModel):
            outputs = self.base_model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # This is a tuple of hidden states from all layers; take the last one.
            hidden = outputs.hidden_states[-1]
        else:
            hidden = self.base_model(input_ids, return_hidden=True)
        pooled = hidden.mean(dim=1)
        reward = self.scorer(pooled)
        return reward.squeeze(-1)



    def loss(self, chosen_reward, rejected_reward):
        # Pairwise ranking loss: chosen should be higher than rejected
        return -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
