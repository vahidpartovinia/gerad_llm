import json

if __name__ == "__main__":
    import psutil
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import RewardTrainer, RewardConfig
    import os
    from model_classification_rlhf import GPT, GPTConfig, RewardModel
    from datasets import load_dataset
    from safetensors.torch import safe_open

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    dataset = load_dataset("Anthropic/hh-rlhf")
    train_subset = dataset["train"].select(range(10000))
    test_subset = dataset["test"].select(range(750))

    resume_from_reward = False
    use_mmlu_checkpoint = True

    config = GPTConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=12,
        n_head=8,
        n_embd=768,
        is_reward_model=True,
        dropout=0.3,
        bias=True
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if resume_from_reward:
        print("Resuming from reward model checkpoint...")
        resume_from_checkpoint = ("./reward_model_test/checkpoint-5894")
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

        print("Creating reward model from scratch or MMLU checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # First create the base model configuration
        config = GPTConfig(
            vocab_size=50257,
            block_size=256,
            n_layer=12,
            n_head=8,
            n_embd=768,
            is_reward_model=True,
            dropout=0.3,
            bias=True
        )
        base_model = GPT(config)

        if use_mmlu_checkpoint:
            print("Loading weights from MMLU checkpoint...")
            # Load MMLU checkpoint
            mmlu_checkpoint = torch.load("out-mmlu/ckpt.pt", map_location='cpu')
            mmlu_state_dict = mmlu_checkpoint['model']

            # DEBUG: Print MMLU state dict keys
            print("MMLU state dict keys:", list(mmlu_state_dict.keys())[:5])

            # Filter out classification head and other non-transformer weights
            transformer_state_dict = {
                k[len("transformer."):]: v
                for k, v in mmlu_state_dict.items()
                if k.startswith('transformer.') and not k.startswith('transformer.classification_head')
            }

            # DEBUG: Print filtered keys
            print("Filtered keys:", list(transformer_state_dict.keys())[:5])

            # Load into the base model's transformer
            base_model.transformer.load_state_dict(transformer_state_dict, strict=False)
            print("Loaded MMLU checkpoint weights into reward model")
            resume_from_checkpoint = None

    model = RewardModel(base_model)

    # Set tokenizer and model config
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",  # Must match original
        target_modules=["c_attn", "c_proj", "c_fc"]
    )

    training_args = RewardConfig(
        output_dir="reward_model",
        num_train_epochs=9,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=0.1,
        max_grad_norm=1.0,

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


    # Create trainer - use model directly
    trainer = RewardTrainer(
        model=model,  # Use base model directly
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
        output = model(**sample_input)
        print("Sample output:", output)

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # trainer.train()


    # Evaluation
    print("Starting evaluation...")
    eval_stats = trainer.evaluate()
    output_dir = "reward_model"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_eval_stats.json")
    with open(output_path, "w") as f:
        json.dump(eval_stats, f, indent=2)