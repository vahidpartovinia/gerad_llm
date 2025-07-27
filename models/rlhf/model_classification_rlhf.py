"""
Fixed GPT Language Model for reward training
Key fixes:
1. Always create lm_head for proper weight loading from pretrained models
2. Improved create_reward_model_from_pretrained function
3. Better weight transfer logic
"""

import math
import inspect
import traceback
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PretrainedConfig, GenerationConfig


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout if self.training else 0,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig(PretrainedConfig):
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    num_classes: int = None
    # Reward model parameters
    is_reward_model: bool = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))

        # ALWAYS create lm_head for proper weight loading from pretrained models
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Different heads for different tasks
        if config.num_classes is not None:
            self.classification_head = nn.Linear(config.n_embd, config.num_classes)

        if config.is_reward_model:
            # Reward head outputs a scalar score
            self.reward_head = nn.Linear(config.n_embd, 1)

        # Weight tying for language modeling (only when not reward model)
        if not config.is_reward_model:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # For GPT-style models, just return the inputs as a dict
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            **kwargs
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if input_ids is not None:
            # Check if input_ids is actually embeddings (float) instead of token IDs (int)
            if input_ids.dtype == torch.float32 and len(input_ids.shape) == 3:
                print("[DEBUG] Converting float32 3D input_ids to inputs_embeds")
                inputs_embeds = input_ids
                input_ids = None

                device = inputs_embeds.device
                b, t, _ = inputs_embeds.size()

                # Skip token embedding, use the provided embeddings directly
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(inputs_embeds + pos_emb)
            else:
                # Normal token ID processing
                device = input_ids.device
                b, t = input_ids.size()
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

                pos = torch.arange(0, t, dtype=torch.long, device=device)
                tok_emb = self.transformer.wte(input_ids)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(tok_emb + pos_emb)
        elif inputs_embeds is not None:
            # Handle direct inputs_embeds case
            device = inputs_embeds.device
            b, t, _ = inputs_embeds.size()

            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(inputs_embeds + pos_emb)
        else:
            raise ValueError("Must provide either input_ids or inputs_embeds")

        # Continue with the same transformer processing for both paths
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        output_hidden_states = kwargs.get('output_hidden_states', False)
        return_dict = kwargs.get('return_dict', True)

        if self.config.is_reward_model:
            if attention_mask is not None:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                sequence_lengths = torch.clamp(sequence_lengths, 0, t - 1)
                batch_indices = torch.arange(b, device=device)
                last_hidden = x[batch_indices, sequence_lengths]
            else:
                last_hidden = x[:, -1, :]

            rewards = self.reward_head(last_hidden).squeeze(-1)

            class RewardModelOutput:
                def __init__(self, rewards, hidden_states):
                    self.rewards = rewards
                    self.hidden_states = hidden_states

            return RewardModelOutput(rewards, [x])


        else:

            targets = kwargs.get('labels') or kwargs.get('targets')

            if targets is not None:

                logits = self.lm_head(x)

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

                result = {"logits": logits, "loss": loss}

            else:

                logits = self.lm_head(x)

                result = {"logits": logits}

            # Always add hidden_states when output_hidden_states=True

            if output_hidden_states:
                result["hidden_states"] = [x]  # List of hidden states from all layers

            # Convert to object if return_dict=True

            # if return_dict:
            #     class GPTOutput:
            #         def __init__(self, **kwargs):
            #             for k, v in kwargs.items():
            #                 setattr(self, k, v)
            #
            #         def __getitem__(self, key):
            #             """Support dict-style access for compatibility"""
            #             return getattr(self, key)
            #
            #         def __contains__(self, key):
            #             """Support 'in' operator"""
            #             return hasattr(self, key)
            #
            #         def keys(self):
            #             """Support .keys() method"""
            #             return [attr for attr in dir(self) if not attr.startswith('_')]

            from transformers.modeling_outputs import CausalLMOutput

            if return_dict:
                return CausalLMOutput(
                    logits=result["logits"],
                    loss=result.get("loss"),
                    hidden_states=result.get("hidden_states")
                )

            return result

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k in ['dropout', 'num_classes', 'is_reward_model'] for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints

        # Apply overrides
        for key, value in override_args.items():
            if key == 'dropout':
                print(f"overriding dropout rate to {value}")
            config_args[key] = value

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # Filter out keys that don't exist in our model (like reward_head for language models)
        sd_keys_filtered = [k for k in sd_keys if k in sd_keys_hf]
        sd_keys_hf_filtered = [k for k in sd_keys_hf if k in sd_keys]

        assert len(sd_keys_hf_filtered) == len(
            sd_keys_filtered), f"mismatched keys: {len(sd_keys_hf_filtered)} != {len(sd_keys_filtered)}"

        for k in sd_keys_hf_filtered:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, **kwargs):
        eos_token_id = 50256
        all_logits = []

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            outputs = self(idx_cond)

            # Handle both dict and object formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits  # Object format
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]  # Dict format
            else:
                logits = outputs  # Raw tensor

            logits = logits[:, -1, :] / temperature

            # Store logits before sampling
            all_logits.append(logits)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if (idx_next == eos_token_id).all():
                break

            idx = torch.cat((idx, idx_next), dim=1)

        class GenerateOutput:
            def __init__(self, sequences, scores):
                self.sequences = sequences
                self.scores = scores

        return GenerateOutput(idx, all_logits)


# Wrapper class for TRL compatibility
class RewardModel(nn.Module):
    """
    Wrapper to ensure full compatibility with TRL RewardTrainer
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.base_model_prefix = "model"

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)

        if hasattr(outputs, 'rewards'):
            return outputs

        if isinstance(outputs, dict) and "rewards" in outputs:
            rewards = outputs["rewards"]
            # Ensure rewards shape is (batch, num_choices, 1)
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(0).unsqueeze(-1)  # (1, batch, 1)

            elif rewards.dim() == 2:
                rewards = rewards.unsqueeze(-1)  # (batch, num_choices, 1)

            class RewardOutput:
                def __init__(self, rewards, hidden_states):
                    self.rewards = rewards
                    self.logits = rewards
                    self.hidden_states = hidden_states

                def __getitem__(self, key):
                    if key == "rewards":
                        return self.rewards
                    if key == "logits":
                        return self.logits
                    raise KeyError(key)

            hidden_states = outputs.get("hidden_states", [rewards])

            return RewardOutput(rewards, hidden_states)

        return outputs

    def score(self, hidden_states):
        """Score method that returns 2D tensor"""
        print(f"[DEBUG] RewardModel.score called with shape: {hidden_states.shape}")

        # Use the model's forward method instead of accessing reward_head directly
        # We need to convert hidden_states back to a format the model expects
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Create dummy input_ids (we'll use inputs_embeds instead)
        device = hidden_states.device
        dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

        # Call the model with inputs_embeds
        outputs = self.model(
            input_ids=None,
            inputs_embeds=hidden_states,
            return_dict=True
        )

        # Extract rewards and repeat for each position
        if hasattr(outputs, 'rewards'):
            rewards = outputs.rewards  # [batch_size]
            # Expand to [batch_size, seq_len] as expected by get_reward
            rewards = rewards.unsqueeze(1).expand(-1, seq_len)
        else:
            raise ValueError("Model output doesn't have rewards")

        print(f"[DEBUG] RewardModel.score returning shape: {rewards.shape}")
        return rewards

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing if needed"""
        pass

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing if needed"""
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)


def create_reward_model_from_pretrained(model_type="gpt2", **kwargs):
    """
    Improved function to create a reward model from pretrained GPT
    """
    print(f"Creating reward model from {model_type}")

    # First, load the pretrained language model
    base_model = GPT.from_pretrained(model_type, override_args=kwargs)

    # Now create the reward model with the same architecture
    reward_config = GPTConfig(
        block_size=base_model.config.block_size,
        vocab_size=base_model.config.vocab_size,
        n_layer=base_model.config.n_layer,
        n_head=base_model.config.n_head,
        n_embd=base_model.config.n_embd,
        dropout=base_model.config.dropout,
        bias=base_model.config.bias,
        is_reward_model=True
    )

    reward_model = GPT(reward_config)

    # Copy ALL transformer weights from the pretrained model
    print("Copying transformer weights...")
    reward_model.transformer.load_state_dict(base_model.transformer.state_dict())

    # Initialize reward head with small random weights
    print("Initializing reward head...")
    nn.init.normal_(reward_model.reward_head.weight, mean=0.0, std=0.02)
    if reward_model.reward_head.bias is not None:
        nn.init.zeros_(reward_model.reward_head.bias)

    print("Reward model created successfully!")
    return RewardModel(reward_model)
