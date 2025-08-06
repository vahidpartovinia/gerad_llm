import os
import gc
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig
from peft import PeftModel

# Import your custom models
from models.rlhf.model_rlhf import GPT

# Environment setup
os.environ["WANDB_MODE"] = "disabled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def clear_memory():
    """Clear memory for all device types"""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class SimplePromptDataset(Dataset):
    """Simple, robust dataset for prompts"""

    def __init__(self, prompts, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Filter and prepare prompts
        self.prompts = []
        for prompt in prompts:
            if prompt and len(prompt.strip()) > 10:  # Only keep reasonable prompts
                self.prompts.append(prompt.strip()[:200])  # Limit length

        print(f"Dataset created with {len(self.prompts)} prompts")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


class SimpleValueHeadModel(torch.nn.Module):
    """Simple wrapper that adds a value head to GPT"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.v_head = torch.nn.Linear(base_model.config.n_embd, 1, bias=False)

        # Add required attributes for PPO
        self.base_model_prefix = "base_model"

        # Add generation config
        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(
            eos_token_id=50256,
            pad_token_id=50256,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )

        # Initialize value head
        torch.nn.init.normal_(self.v_head.weight, std=0.02)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # For training, we need full sequence logits, so call with targets=None but get full sequence
        device = input_ids.device
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward the GPT model to get full sequence (copied from shakespeare model)
        tok_emb = self.base_model.transformer.wte(input_ids)
        pos_emb = self.base_model.transformer.wpe(pos)
        x = self.base_model.transformer.drop(tok_emb + pos_emb)
        for block in self.base_model.transformer.h:
            x = block(x)
        x = self.base_model.transformer.ln_f(x)

        # Get full sequence logits (not just last token)
        logits = self.base_model.lm_head(x)  # Shape: [batch, seq_len, vocab_size]

        # Get values (use last token hidden state)
        values = self.v_head(x[:, -1, :]).squeeze(-1)

        # Return in expected format
        class ModelOutput:
            def __init__(self, logits, value):
                self.logits = logits
                self.value = value

        return ModelOutput(logits, values)

    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_k=None, **kwargs):
        return self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable()


def load_simple_dataset(max_samples):
    """Load a small subset of HH-RLHF data"""
    print("Loading dataset...")

    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")

        prompts = []
        for i, example in enumerate(dataset):
            if i >= max_samples:
                break

            # Extract human message
            conversation = example['chosen']
            parts = conversation.split('\n\nHuman: ')

            if len(parts) > 1:
                human_part = parts[1].split('\n\nAssistant:')[0].strip()
                if human_part and len(human_part) > 20:
                    prompts.append(human_part)

        print(f"Extracted {len(prompts)} prompts")
        return prompts

    except Exception as e:
        print(f"Failed to load HH-RLHF: {e}")
        # Fallback to dummy data
        return [
            "What is the capital of France?",
            "How do I make a sandwich?",
            "Explain quantum physics simply.",
            "What's the weather like?",
            "Tell me a joke."
        ]


def create_policy_model():
    """Create policy model for generation"""
    print("Creating policy model...")

    # Use the language modeling GPT, not classification
    base_model = GPT.from_pretrained("gpt2")
    policy_model = SimpleValueHeadModel(base_model)

    return policy_model


def load_reward_model(checkpoint_path):
    """Load reward model with proper error handling"""
    try:
        print(f"Loading reward model from: {checkpoint_path}")

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint path does not exist: {checkpoint_path}")
            return None

        # List files in checkpoint
        files = os.listdir(checkpoint_path)
        print(f"Files in checkpoint: {files}")

        # 1. Create the base model
        from models.shakespeare.model import GPT, GPTConfig

        config = GPTConfig(
            vocab_size=50257,
            block_size=256,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.3,
            bias=True
        )

        # Add missing attributes for PEFT compatibility
        config.model_type = "gpt"
        config.tie_word_embeddings = False

        def config_get(self, key, default=None):
            return getattr(self, key, default)

        config.get = config_get.__get__(config, type(config))

        # 2. Create base model
        base_model = GPT(config)

        # Add missing methods for PEFT compatibility
        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

        def can_generate(self):
            return True

        base_model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(base_model, GPT)
        base_model.can_generate = can_generate.__get__(base_model, GPT)

        # 3. Load the PEFT adapters
        print("Loading PEFT adapters...")
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

        # 4. Create RewardModel wrapper
        class RewardModel(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                self.config = base_model.config

                # Add reward head
                self.reward_head = torch.nn.Linear(base_model.config.n_embd, 1, bias=False)
                torch.nn.init.normal_(self.reward_head.weight, std=0.02)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                # Get hidden states by running through transformer manually
                device = input_ids.device
                b, t = input_ids.size()
                pos = torch.arange(0, t, dtype=torch.long, device=device)

                # Access the actual base model through PEFT wrapper
                if hasattr(self.base_model, 'base_model'):
                    transformer = self.base_model.base_model.transformer
                else:
                    transformer = self.base_model.transformer

                tok_emb = transformer.wte(input_ids)
                pos_emb = transformer.wpe(pos)
                x = transformer.drop(tok_emb + pos_emb)

                for block in transformer.h:
                    x = block(x)
                x = transformer.ln_f(x)

                # Get reward score using last token
                rewards = self.reward_head(x[:, -1, :]).squeeze(-1)

                return {
                    "rewards": rewards,
                    "end_scores": rewards
                }

            def score(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.forward(input_ids, attention_mask=attention_mask, **kwargs)
                return outputs["rewards"]

        # 5. Create the reward model with PEFT adapters
        reward_model = RewardModel(peft_model)
        reward_model.eval()

        print("Successfully loaded reward model")
        return reward_model

    except Exception as e:
        print(f"Failed to load reward model: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_dummy_reward_model():
    """Fallback dummy reward model"""

    class DummyRewardModel(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            return torch.randn(input_ids.shape[0], device=input_ids.device) * 0.1

        def score(self, input_ids, **kwargs):
            return self.forward(input_ids, **kwargs)

    return DummyRewardModel()


def train_ppo():
    """Main training function"""
    print("Starting PPO training...")
    clear_memory()

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    prompts = load_simple_dataset(max_samples=1500)
    dataset = SimplePromptDataset(prompts, tokenizer, max_length=128)

    # Test dataset
    print("Testing dataset...")
    sample = dataset[0]
    print(f"Sample shape: {sample['input_ids'].shape}")

    # Create models
    policy_model = create_policy_model().to(device)
    reward_model = load_reward_model("reward_model_shakespeare/checkpoint-5871").to(device)

    clear_memory()

    # ---------------------------------------------------------------------------------------------------------

    print("Using manual PPO-style training...")

    # Manual PPO-style training loop
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=2e-6)  # Increased from 1e-6
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_loss = float('inf')
    best_reward = float('-inf')

    best_model_dir = "best_ppo_model"

    continue_from_checkpoint = True
    # Load checkpoint if requested
    if continue_from_checkpoint and os.path.exists("best_ppo_model/best_model.pt"):
        print("Loading checkpoint from best_ppo_model")
        checkpoint = torch.load("best_ppo_model/best_model.pt", map_location=device)

        policy_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.3f}")
    else:
        print("Starting fresh training")

    clear_memory()

    num_epochs = 50
    for epoch in range(num_epochs):
        if continue_from_checkpoint:
            print(f"\n=== Epoch {start_epoch + 1}/{start_epoch + num_epochs} ===")
        else:
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        epoch_rewards = []
        epoch_losses = []

        for step, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):

            if step >= 10:
                break

            try:
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Generate responses
                with torch.no_grad():
                    generation_output = policy_model.generate(
                        input_ids,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )

                    if hasattr(generation_output, 'sequences'):
                        response_ids = generation_output.sequences
                    else:
                        response_ids = generation_output

                # Get rewards
                with torch.no_grad():
                    reward_output = reward_model.score(response_ids)

                    if hasattr(reward_output, 'rewards'):
                        rewards = reward_output.rewards
                    elif hasattr(reward_output, 'logits'):
                        rewards = reward_output.logits
                    else:
                        rewards = reward_output

                    if not isinstance(rewards, torch.Tensor):
                        rewards = torch.tensor(rewards, device=response_ids.device)

                    if rewards.dim() > 1:
                        rewards = rewards.squeeze()
                        if rewards.dim() > 1:
                            rewards = rewards.mean(dim=-1)

                # Simple policy gradient update
                optimizer.zero_grad()

                # Forward pass to get log probabilities
                model_output = policy_model(response_ids)
                logits = model_output.logits
                values = model_output.value

                log_probs = torch.log_softmax(logits, dim=-1)
                selected_log_probs = torch.gather(log_probs, -1, response_ids.unsqueeze(-1)).squeeze(-1)
                selected_log_probs = torch.clamp(selected_log_probs, min=-10, max=0)

                advantages = rewards - values.detach()

                # Normalize advantages for more stable training
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_loss = -(selected_log_probs * advantages).mean()

                # Use Huber loss for value function (more stable)
                value_loss = torch.nn.functional.huber_loss(values, rewards)

                # Add KL penalty to prevent policy from changing too quickly
                kl_penalty = 0.01 * torch.nn.functional.kl_div(
                    torch.log_softmax(logits, dim=-1),
                    torch.softmax(logits.detach(), dim=-1),
                    reduction='batchmean'
                )

                total_loss = policy_loss + 0.1 * value_loss + kl_penalty
                loss_val = total_loss.item()

                # Track stats
                avg_reward = rewards.mean().item()
                loss_val = total_loss.item()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                epoch_rewards.append(avg_reward)
                epoch_losses.append(loss_val)

                # Print progress every few steps
                if step % 2 == 0 or step == 9:
                    print(
                        f"Step {step + 1:2d}/10 | Reward: {avg_reward:6.3f} | Loss: {loss_val:6.3f} | Policy Loss: {policy_loss.item():6.3f} | Value Loss: {value_loss.item():6.3f}")

                # Clear memory periodically
                if step % 2 == 0:
                    clear_memory()

            except Exception as e:
                print(f"Error in step {step}: {e}")
                continue

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        save_from_loss = False  # Set to True to save when loss is lowest
        save_from_reward = True  # Set to True to save when reward is highest

        if epoch_rewards:
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

            # Determine if we should save based on configuration
            should_save = False
            save_reason = ""

            if save_from_reward and avg_epoch_reward > best_reward:
                best_reward = avg_epoch_reward
                should_save = True
                save_reason = f"New best reward: {best_reward:.3f}"

            if save_from_loss and avg_epoch_loss < best_loss and avg_epoch_loss > 0:
                best_loss = avg_epoch_loss
                should_save = True
                if save_reason:
                    save_reason = f"New best reward: {best_reward:.3f} and loss: {best_loss:.3f}"
                else:
                    save_reason = f"New best loss: {best_loss:.3f}"

            if should_save:
                print(f"{save_reason} - Saving model...")

                # Save the best model
                os.makedirs(best_model_dir, exist_ok=True)
                torch.save({
                    'model_state_dict': policy_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_reward': best_reward,
                    'best_loss': best_loss if save_from_loss else None,
                    'avg_epoch_reward': avg_epoch_reward,
                    'avg_epoch_loss': avg_epoch_loss
                }, f"{best_model_dir}/best_model.pt")

                tokenizer.save_pretrained(best_model_dir)
                print(f"Best model saved to {best_model_dir}")

            print(
                f"Epoch {epoch + 1} Summary | Avg Reward: {avg_epoch_reward:.3f} | Avg Loss: {avg_epoch_loss:.3f} | LR: {current_lr:.2e}")
        else:
            print(f"Epoch {epoch + 1} failed - no successful steps")

    print("Training completed!")

    # Save model
    save_dir = "./simple_ppo_model"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), f"{save_dir}/policy_model.pt")
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    train_ppo()