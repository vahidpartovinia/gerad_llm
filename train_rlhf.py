if __name__ == "__main__":
    import psutil
    import torch
    from peft import LoraConfig, TaskType, PeftModel
    from transformers import AutoTokenizer
    from trl import RewardTrainer, RewardConfig
    import os
    from model_classification_rlhf import GPT, GPTConfig
    from datasets import load_dataset
    import random

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    dataset = load_dataset("Anthropic/hh-rlhf")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_subset = train_ds.select(range(2000))  # Use first 4000 examples
    test_subset = test_ds.select(range(750))

    print("Chosen: ", train_subset[1]['chosen'], " end")
    print("Rejected: ",train_subset[1]['rejected'], " end")

    config = GPTConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=12,
        n_head=8,
        n_embd=768,
        num_classes=1
    )
    model = GPT(config)

    resume_from_reward = False  # Set to False to start from MMLU

    if resume_from_reward:
        print("Resuming from reward model checkpoint...")
        resume_from_checkpoint = "./reward_model/checkpoint-400"
        tokenizer = AutoTokenizer.from_pretrained("./reward_model/checkpoint-400", local_files_only=True)
    else:
        print("Starting from MMLU checkpoint...")
        # Start from MMLU checkpoint
        checkpoint = torch.load("out-mmlu/ckpt.pt", map_location='mps')
        state_dict = checkpoint['model']
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classification_head')}
        model.load_state_dict(state_dict, strict=False)
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
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
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

    def fixed_compute_metrics(eval_pred):
        return {}  # Return empty dict to skip metrics computation

    RewardTrainer.visualize_samples = fixed_visualize_samples
    RewardTrainer.compute_metrics = fixed_compute_metrics

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_subset,  # Use subset instead of full dataset
        eval_dataset=test_subset,
        peft_config=peft_config,
    )

    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    if torch.backends.mps.is_available():
        print("Using MPS device")

    trainer.train()