import gc
import json

if __name__ == "__main__":
    import psutil
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import RewardTrainer, RewardConfig
    import os
    from models.shakespeare.model import GPT, GPTConfig  # Changed import
    from datasets import load_dataset
    from safetensors.torch import safe_open

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    dataset = load_dataset("Anthropic/hh-rlhf")
    train_subset = dataset["train"].select(range(10000))
    test_subset = dataset["test"].select(range(750))

    resume_from_reward = False
    use_mmlu_checkpoint = False


    def clear_memory():
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def convert_mmlu_to_shakespeare():
        """Convert MMLU classification checkpoint to shakespeare model"""
        print("Converting MMLU checkpoint to shakespeare architecture...")

        # Load MMLU checkpoint
        mmlu_checkpoint = torch.load("../../out-mmlu/ckpt.pt", map_location='cpu')
        mmlu_state_dict = mmlu_checkpoint['model']

        print("MMLU checkpoint keys (first 5):", list(mmlu_state_dict.keys())[:5])

        # Create shakespeare model
        config = GPTConfig(
            vocab_size=50257,
            block_size=256,  # Match your training config
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.3,
            bias=True
        )
        shakespeare_model = GPT(config)

        # Extract transformer weights (excluding classification head)
        transformer_weights = {}
        for k, v in mmlu_state_dict.items():
            if k.startswith('transformer.') and 'classification_head' not in k:
                # Remove 'transformer.' prefix for shakespeare model
                new_key = k[len('transformer.'):]
                transformer_weights[new_key] = v

        print("Converted weights keys (first 5):", list(transformer_weights.keys())[:5])

        # Load weights into shakespeare model
        missing_keys, unexpected_keys = shakespeare_model.transformer.load_state_dict(transformer_weights, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        print("✅ Successfully converted MMLU to shakespeare architecture")
        return shakespeare_model

    # Create RewardModel class for shakespeare model
    class RewardModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.config = base_model.config

            # Add reward head
            self.reward_head = torch.nn.Linear(base_model.config.n_embd, 1, bias=False)
            torch.nn.init.normal_(self.reward_head.weight, std=0.02)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            # Get base model outputs
            logits, _ = self.base_model(input_ids)

            # Get hidden states by running through transformer manually
            device = input_ids.device
            b, t = input_ids.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)

            tok_emb = self.base_model.transformer.wte(input_ids)
            pos_emb = self.base_model.transformer.wpe(pos)
            x = self.base_model.transformer.drop(tok_emb + pos_emb)

            for block in self.base_model.transformer.h:
                x = block(x)
            x = self.base_model.transformer.ln_f(x)

            # Get reward score using last token
            rewards = self.reward_head(x[:, -1, :]).squeeze(-1)

            # Return both logits and rewards for TRL compatibility
            return {
                "logits": logits,
                "rewards": rewards,
                "end_scores": rewards  # TRL also expects this
            }

        def score(self, input_ids, attention_mask=None, **kwargs):
            outputs = self.forward(input_ids, attention_mask=attention_mask, **kwargs)
            return outputs["rewards"]

        # Add missing methods for TRL compatibility
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

        def get_input_embeddings(self):
            return self.base_model.transformer.wte

        def set_input_embeddings(self, new_embeddings):
            self.base_model.transformer.wte = new_embeddings

        def get_output_embeddings(self):
            return None  # Reward model doesn't have output embeddings

        def set_output_embeddings(self, new_embeddings):
            pass  # Reward model doesn't have output embeddings

        def resize_token_embeddings(self, new_num_tokens):
            return self.get_input_embeddings()

        def tie_weights(self):
            pass  # No weights to tie in reward model


    config = GPTConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=12,
        n_head=12,  # Keep standard GPT-2 config
        n_embd=768,
        dropout=0.3,
        bias=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if resume_from_reward:
        print("Resuming from reward model checkpoint...")
        resume_from_checkpoint = ("./reward_model_shakespeare/checkpoint-1957")  # Update path
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint, local_files_only=True)

        # Create base model
        base_model = GPT(config)

        # Load checkpoint directly into base model
        safetensors_path = f"{resume_from_checkpoint}/adapter_model.safetensors"
        state_dict = {}
        with safe_open(safetensors_path, framework="pt", device="mps") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        base_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {safetensors_path}")
    else:

        print("Creating reward model from MMLU checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Load MMLU checkpoint and convert to shakespeare architecture
        base_model = convert_mmlu_to_shakespeare()
        resume_from_checkpoint = None

    model = RewardModel(base_model)

    def config_get(self, key, default=None):
        return getattr(self, key, default)


    def config_to_dict(self):
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'bias': self.bias,
            'tie_word_embeddings': getattr(self, 'tie_word_embeddings', True),
            'pad_token_id': getattr(self, 'pad_token_id', 50256),
            'model_type': 'gpt2'
        }


    # Bind the methods to the config
    model.config.get = config_get.__get__(model.config, type(model.config))
    model.config.to_dict = config_to_dict.__get__(model.config, type(model.config))

    # Add missing attributes
    if not hasattr(model.config, 'tie_word_embeddings'):
        model.config.tie_word_embeddings = True
    if not hasattr(model.config, 'model_type'):
        model.config.model_type = 'gpt2'

    # Set tokenizer and model config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"]
    )

    training_args = RewardConfig(
        output_dir="reward_model_shakespeare",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        warmup_steps=200,
        weight_decay=0.1,
        max_grad_norm=10.0,

        logging_dir='./logs',
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy="steps",
        save_steps=500,
        eval_steps=250,

        use_mps_device=True,
        bf16=False,
        max_length=256,
        resume_from_checkpoint=resume_from_checkpoint if resume_from_reward else None,
        save_safetensors=True,
    )

    # Monkey-patch to avoid visualization errors
    from trl.trainer.reward_trainer import RewardTrainer


    def fixed_visualize_samples(self, num_print_samples):
        pass


    RewardTrainer.visualize_samples = fixed_visualize_samples


    def compute_eval_metrics(eval_pred):
        predictions, _ = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        import numpy as np
        rewards = np.array(predictions).squeeze()
        return {"mean_reward": float(np.mean(rewards))}


    # Create trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_subset,
        eval_dataset=test_subset,
        peft_config=peft_config,
        compute_metrics=compute_eval_metrics,
    )

    print(f"Available RAM: {psutil.virtual_memory().available / 1024 ** 3:.1f} GB")
    if torch.backends.mps.is_available():
        print("Using MPS device")
        device = torch.device("mps")
        model.to(device)

    # Debug: Print model architecture
    print("Model architecture:", model)
    print("Model device:", next(model.parameters()).device)

    # Debug: Print sample output before training
    with torch.no_grad():
        sample_input = tokenizer("Sample input text", return_tensors="pt").to(device)
        input_ids = sample_input["input_ids"]
        output = model(input_ids)
        print("Sample output:", output)

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    clear_memory()

    # Evaluation
    print("Starting evaluation...")
    eval_stats = trainer.evaluate()
    output_dir = "reward_model_shakespeare"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_eval_stats.json")
    with open(output_path, "w") as f:
        json.dump(eval_stats, f, indent=2)