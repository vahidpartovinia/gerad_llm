import json

if __name__ == "__main__":
    import psutil
    import torch
    from peft import LoraConfig, TaskType, PeftModel
    from transformers import AutoTokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel
    from trl import RewardTrainer, RewardConfig
    import os
    from model_classification_rlhf import create_reward_model_from_pretrained, GPT, GPTConfig, RewardModel
    from datasets import load_dataset


    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    dataset = load_dataset("Anthropic/hh-rlhf")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_subset = train_ds.select(range(2000))  # Use first 4000 examples
    test_subset = test_ds.select(range(750))

    # Create config for the reward model
    config = GPTConfig(
        vocab_size=50257,
        block_size=256,  # or whatever you used
        n_layer=12,
        n_head=8,
        n_embd=768,
        is_reward_model=True,  # This is crucial!
        dropout=0.1,
        bias=True
    )

    # Create model and load checkpoint
    model = GPT(config)

    resume_from_reward = True  # Set to True to resume from reward checkpoint

    if resume_from_reward:
        print("Resuming from reward model checkpoint...")

        # Load the saved reward model
        resume_from_checkpoint = "./reward_model/checkpoint-990"

        # checkpoint = torch.load(f"{resume_from_checkpoint}/pytorch_model.bin", map_location='cpu')
        # model.load_state_dict(checkpoint, strict=False)

        # Wrap in RewardModel for TRL compatibility
        from model_classification_rlhf import RewardModel
        reward_model = RewardModel(model)
        tokenizer = AutoTokenizer.from_pretrained(resume_from_checkpoint, local_files_only=True)

    else:
        print("Creating reward model from pretrained/MMLU checkpoint...")

        # First create from pretrained, then load MMLU weights
        if False:  # Set to True if you want to use MMLU checkpoint
            # Load MMLU checkpoint
            mmlu_checkpoint = torch.load("out-mmlu/ckpt.pt", map_location='cpu')
            mmlu_state_dict = mmlu_checkpoint['model']

            # Filter out classification head and other non-transformer weights
            transformer_state_dict = {
                k: v for k, v in mmlu_state_dict.items()
                if k.startswith('transformer.') and not k.startswith('classification_head')
            }

            # Load into the reward model's transformer
            reward_model = RewardModel(model)
            reward_model.model.transformer.load_state_dict(transformer_state_dict, strict=False)
            print("Loaded MMLU checkpoint weights into reward model")
            resume_from_checkpoint = None
            tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Always set pad token for GPT2
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        r=4,  # Increase adapter rank
        lora_alpha=8,  # Increase scaling
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc", "c_proj"]
    )

    training_args = RewardConfig(
        output_dir="./reward_model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,   # Effective batch size = 64 * 4 = 256
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        weight_decay=0.01,
        max_grad_norm=1.0,

        logging_dir='./logs',
        logging_steps=10,
        logging_strategy="steps",

        save_strategy="epoch",
        eval_strategy="steps",
        save_steps=500,
        eval_steps=2000,

        use_mps_device=True,
        bf16=False,
        dataloader_num_workers=4,        # Parallel data loading

        max_length=256,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    from trl.trainer.reward_trainer import RewardTrainer

    def fixed_visualize_samples(self, num_print_samples):
        pass

    def compute_eval_metrics(eval_pred):
        predictions, labels = eval_pred
        # If predictions is a tuple, get the first element
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        # Convert to numpy if needed
        import numpy as np
        rewards = np.array(predictions).squeeze()
        return {"mean_reward": float(np.mean(rewards))}

    RewardTrainer.visualize_samples = fixed_visualize_samples
    # RewardTrainer.compute_metrics = compute_eval_metrics

    reward_model = RewardModel(model)

    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_subset,  # Use subset instead of full dataset
        eval_dataset=test_subset,
        peft_config=peft_config,
        compute_metrics=compute_eval_metrics,

    )

    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    if torch.backends.mps.is_available():
        print("Using MPS device")

    # trainer.train()

    # -----------------------------------------------------------

    # To evaluate

    # Load base model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "reward_model/checkpoint-990")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("reward_model/checkpoint-990", local_files_only=True)

    print("Starting evaluation...")
    eval_stats = trainer.evaluate()
    output_dir = "./reward_model"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_eval_stats.json")
    with open(output_path, "w") as f:
        json.dump(eval_stats, f, indent=2)
    print(f"Final eval stats saved to {output_path}")