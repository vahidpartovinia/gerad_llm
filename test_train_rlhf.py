
if __name__ == "__main__":
    import psutil
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2ForSequenceClassification
    from trl import RewardTrainer, RewardConfig
    import os
    from datasets import load_dataset
    import torch

    dataset = load_dataset("Anthropic/hh-rlhf")

    train_ds = dataset["train"]
    test_ds = dataset["test"]

    train_subset = train_ds.select(range(1000))  # Use first 4000 examples
    test_subset = test_ds.select(range(750))  # Use first 500 examples

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    #
    # # If using base GPT2:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=1)

    # If using your checkpoint:
    # tokenizer = AutoTokenizer.from_pretrained("reward_model/checkpoint-248")
    # model = GPT2ForSequenceClassification.from_pretrained("reward_model/checkpoint-248", num_labels=1)

    # Always set pad token for GPT2
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    training_args = RewardConfig(
        output_dir="./reward_model",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch size = 64 * 4 = 256
        learning_rate=2e-4,
        logging_dir='./logs',
        logging_steps=10,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=500,
        use_mps_device=True,
        bf16=False,
        dataloader_num_workers=4,  # Parallel data loading
        max_length=512,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_subset,  # Use subset instead of full dataset
        eval_dataset=test_subset,
        peft_config=peft_config,
    )

    print(f"Available RAM: {psutil.virtual_memory().available / 1024 ** 3:.1f} GB")
    if torch.backends.mps.is_available():
        print("Using MPS device")

    trainer.train()
