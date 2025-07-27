"""
Simple PPO Training Script for Custom GPT with Trained Reward Model
"""
import inspect
import sys
import traceback

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import trl
from peft import PeftModel
import os
import json
from model_classification_rlhf import GPT, GPTConfig, RewardModel, create_reward_model_from_pretrained
import trl.trainer.utils

os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device("cpu")


def load_hh_rlhf_dataset(split="train", max_samples=1000):
    """Load and process Anthropic HH-RLHF dataset for PPO training"""

    print(f"Loading HH-RLHF dataset ({split} split)...")

    # Load the dataset
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)

    # Take subset if specified
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    # Extract prompts from the conversations
    prompts = []

    for example in dataset:
        # The dataset has 'chosen' and 'rejected' conversations
        # Extract the human part (prompt) from the chosen conversation
        conversation = example['chosen']

        # Split by "\n\nHuman: " and "\n\nAssistant: "
        parts = conversation.split('\n\nHuman: ')

        if len(parts) > 1:
            # Get the first human message as the prompt
            human_part = parts[1].split('\n\nAssistant:')[0].strip()
            if human_part:  # Only add non-empty prompts
                prompts.append(human_part)

    print(f"Extracted {len(prompts)} prompts from HH-RLHF dataset")
    return prompts


class PromptDataset(Dataset):
    """Simple dataset wrapper for prompts"""

    def __init__(self, prompts):
        self.prompts = prompts
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Tokenize the prompt
        tokenized = self.tokenizer(
            self.prompts[idx],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        return {"input_ids": tokenized["input_ids"].squeeze()}


# Usage in training script:
prompts = load_hh_rlhf_dataset(split="train", max_samples=150)
train_dataset = PromptDataset(prompts)

eval_prompts = load_hh_rlhf_dataset(split="test", max_samples=100)
eval_dataset = PromptDataset(eval_prompts)


def load_reward_model(reward_model_path, base_model_type="gpt2"):
    """Load your trained reward model with adapters"""
    print("Loading reward model...")

    # Load the base reward model
    base_reward_model = create_reward_model_from_pretrained(base_model_type)

    # Load the trained adapter
    peft_model = PeftModel.from_pretrained(base_reward_model, reward_model_path)
    peft_model.eval()

    # Wrap it with RewardModel class for compatibility
    wrapped_model = RewardModel(peft_model)

    return wrapped_model


def create_policy_model(model_type="gpt2", **kwargs):
    """Create policy model (same architecture as base model)"""
    print("Creating policy model...")

    # Load custom GPT model
    base_model = GPT.from_pretrained(model_type, override_args=kwargs)

    # Create value head wrapper manually for custom model
    class CustomValueHeadModel(torch.nn.Module):
        class ModelOutput:
            def __init__(self, logits, value, hidden_states):
                self.logits = logits
                self.value = value
                self.hidden_states = hidden_states

        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            self.config = base_model.config
            self.base_model_prefix = "base_model"

            from transformers import GenerationConfig
            self.generation_config = GenerationConfig(
                eos_token_id=50256,
                pad_token_id=50256,
                max_length=1024,
            )

            # Add value head for PPO
            self.v_head = torch.nn.Linear(base_model.config.n_embd, 1, bias=False)

            # Initialize value head
            torch.nn.init.normal_(self.v_head.weight, mean=0.0, std=0.02)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            print(f"[DEBUG] CustomValueHeadModel id: {id(self)}")
            print(f"[DEBUG] This call will return ModelOutput")

            # Get outputs from base model
            outputs = self.base_model(input_ids, attention_mask=attention_mask, **kwargs)

            # Forward pass through transformer for value head
            device = input_ids.device
            b, t = input_ids.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)

            tok_emb = self.base_model.transformer.wte(input_ids)
            pos_emb = self.base_model.transformer.wpe(pos)
            x = self.base_model.transformer.drop(tok_emb + pos_emb)

            for block in self.base_model.transformer.h:
                x = block(x)
            x = self.base_model.transformer.ln_f(x)

            # Get value predictions (use last token)
            values = self.v_head(x[:, -1, :]).squeeze(-1)

            logits = outputs["logits"] if isinstance(outputs, dict) and "logits" in outputs else outputs

            print(f"[DEBUG] About to create ModelOutput with logits type: {type(logits)}")
            print(f"[DEBUG] ModelOutput class: {self.ModelOutput}")

            result = self.ModelOutput(logits, values, [x])

            print(f"[DEBUG] Created result type: {type(result)}")
            print(f"[DEBUG] Result has hidden_states: {hasattr(result, 'hidden_states')}")

            return result

        def score(self, input_ids, attention_mask=None, **kwargs):
            """Score method for reward computation"""
            print(f"[DEBUG] CustomValueHeadModel.score called with input shape: {input_ids.shape}")
            print(f"[DEBUG] Input dtype: {input_ids.dtype}")

            # The input might be hidden states, not token IDs
            if len(input_ids.shape) == 3 and input_ids.shape[-1] == self.base_model.config.n_embd:
                print("[DEBUG] Input looks like hidden states, using directly for value head")
                # Input is already hidden states, use for all positions
                batch_size, seq_len, hidden_size = input_ids.shape

                # Apply value head to all positions, not just last
                values = self.v_head(input_ids)  # Shape: [batch_size, seq_len, 1]
                values = values.squeeze(-1)  # Shape: [batch_size, seq_len]

                print(f"[DEBUG] Returning values with shape: {values.shape}")
                return values
            else:
                # Normal forward pass
                outputs = self.forward(input_ids, attention_mask=attention_mask, **kwargs)
                return outputs.value

        def generate(self, input_ids, max_new_tokens=100, **kwargs):
            return self.base_model.generate(input_ids, max_new_tokens, **kwargs)

        def gradient_checkpointing_enable(self):
            pass

        def gradient_checkpointing_disable(self):
            pass

    model = CustomValueHeadModel(base_model)
    return model


def simple_ppo_training():
    """Main PPO training function"""
    import torch
    device = torch.device("cpu")

    # Save original function
    original_get_reward = trl.trainer.utils.get_reward

    # 1. Load tokenizer (GPT-2)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load trained reward model
    reward_model_path = "reward_model/checkpoint-4410"
    reward_model = load_reward_model(reward_model_path).to(device)

    # 3. Create policy model (same architecture as reward model base)
    policy_model = create_policy_model("gpt2").to(device)  # Match your reward model's base

    # 4. PPO Configuration
    ppo_config = PPOConfig(
        learning_rate=5e-6,
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        num_ppo_epochs=1,
        seed=42,
        remove_unused_columns=False,
        bf16=False,
    )

    value_model = create_policy_model("gpt2").to(device)  # Same as policy model
    value_model.base_model_prefix = "base_model"  # Add required attributes

    # # Add debugging to value model too
    # original_forward = value_model.forward
    #
    # def debug_forward(self, *args, **kwargs):
    #     # print(f"[DEBUG] Value model forward called, id: {id(self)}")
    #     result = original_forward(*args, **kwargs)
    #     # print(f"[DEBUG] Value model returning: {type(result)}")
    #     return result

    # value_model.forward = debug_forward.__get__(value_model, value_model.__class__)


    # 5. Create PPO trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        ref_model=None,  # Will create reference model automatically
        reward_model=reward_model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        value_model=value_model,
        eval_dataset=eval_dataset,
    )

    checkpoint_dir = "./ppo_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)


    # 7. Training loop
    print("Starting PPO training...")

    def load_checkpoint(checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"Loading checkpoint from {checkpoint_path}")

        try:
            # Load model weights
            if os.path.exists(f"{checkpoint_path}/model_weights.pt"):
                model_state = torch.load(f"{checkpoint_path}/model_weights.pt")
                ppo_trainer.model.load_state_dict(model_state)
                print("Model weights loaded successfully")

            # Load training state
            if os.path.exists(f"{checkpoint_path}/training_state.pt"):
                training_state = torch.load(f"{checkpoint_path}/training_state.pt")
                print(f"Resuming from step {training_state['current_step']}")
                return training_state["current_step"]

        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return 0

        return 0

    def save_checkpoint(step):
        """Save checkpoint at current step with shared tensor handling"""
        step_dir = f"{checkpoint_dir}/step_{step}"
        os.makedirs(step_dir, exist_ok=True)

        # Define training_state here so it's always available
        training_state = {
            "current_step": step,
            "completed_batches": step
        }

        try:
            # Clear memory first
            torch.mps.empty_cache()

            # Try to save model weights only (avoid shared tensor issues)
            model_state = {k: v.clone() for k, v in ppo_trainer.model.state_dict().items()}
            torch.save(model_state, f"{step_dir}/model_weights.pt")
            torch.save(training_state, f"{step_dir}/training_state.pt")

            print(f"Checkpoint saved at step {step}")

        except Exception as save_error:
            print(f"Failed to save checkpoint: {save_error}")

            # Minimal save - just the step info
            try:
                torch.save(training_state, f"{step_dir}/training_state.pt")
                print(f"At least saved training state at step {step}")
            except Exception as minimal_error:
                print(f"Even minimal save failed: {minimal_error}")


    global_step = 0
    checkpoint_dir = "./ppo_checkpoints"

    # Check for existing checkpoints
    if os.path.exists(checkpoint_dir):
        # Find the latest checkpoint
        checkpoint_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('step_')]
        if checkpoint_dirs:
            steps = [int(d.split('_')[1]) for d in checkpoint_dirs]
            latest_step = max(steps)
            latest_checkpoint = f"{checkpoint_dir}/step_{latest_step}"

            print(f"Found checkpoint at step {latest_step}")
            global_step = load_checkpoint(latest_checkpoint)
        else:
            print("No checkpoints found, starting fresh")
    else:
        print("No checkpoint directory found, starting fresh")

    try:
        print(f"Starting/resuming training from step {global_step}")
        stats = ppo_trainer.train()
        print(f"Training completed: {stats}")

    except Exception as e:
        print(f"Training crashed at step {global_step}, saving emergency checkpoint...")
        save_checkpoint(global_step)
        raise

    # Final save
    ppo_trainer.save_model("./ppo_trained_model")

    save_dir = "./ppo_trained_model"
    os.makedirs(save_dir, exist_ok=True)

    # Save optimizer state
    torch.save(ppo_trainer.optimizer.state_dict(), f"{save_dir}/optimizer.pt")

    # Save scheduler state if it exists
    if hasattr(ppo_trainer, 'scheduler') and ppo_trainer.scheduler is not None:
        torch.save(ppo_trainer.scheduler.state_dict(), f"{save_dir}/scheduler.pt")

    # Save training config
    ppo_trainer.config.save_pretrained(save_dir)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save current step/epoch info
    training_state = {
        "current_step": ppo_trainer.state.global_step if hasattr(ppo_trainer, 'state') else 0,
        "current_epoch": 1,
    }
    torch.save(training_state, f"{save_dir}/training_state.pt")

    print(f"Training saved to {save_dir}")


if __name__ == "__main__":
    device = torch.device("cpu")
    simple_ppo_training()
