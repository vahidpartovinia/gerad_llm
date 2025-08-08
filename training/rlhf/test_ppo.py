import torch
from transformers import AutoTokenizer
from models.shakespeare.model import GPT
import os

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SimpleValueHeadModel(torch.nn.Module):
    """Same model class from training"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.v_head = torch.nn.Linear(base_model.config.n_embd, 1, bias=False)

        from transformers import GenerationConfig
        self.generation_config = GenerationConfig(
            eos_token_id=50256,
            pad_token_id=50256,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=None, **kwargs):
        return self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )


def load_trained_model(model_path="./best_ppo_model"):
    """Load the trained PPO model"""
    print(f"Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Tokenizer from gpt2 is fine
    tokenizer.pad_token = tokenizer.eos_token

    # Create base model structure (but don't load pretrained weights yet)
    from models.shakespeare.model import GPT, GPTConfig

    # Create config matching GPT-2 architecture
    config = GPTConfig(
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=True
    )

    # Create model from config (not pretrained)
    base_model = GPT(config)
    policy_model = SimpleValueHeadModel(base_model)

    # Load YOUR trained weights
    checkpoint = torch.load(f"{model_path}/best_model.pt", map_location=device)
    policy_model.load_state_dict(checkpoint['model_state_dict'])
    policy_model.eval()

    print(f"Loaded YOUR trained model from epoch {checkpoint['epoch']}")
    if checkpoint['best_loss'] is not None:
        print(f"Best loss: {checkpoint['best_loss']:.3f}")
    else:
        print(f"Best loss: {checkpoint['best_loss']}")
    print(f"Best reward: {checkpoint['best_reward']:.3f}")
    # print(f"Best reward: {checkpoint['reward']:.3f}")
    return policy_model.to(device), tokenizer


def test_model_responses(model, tokenizer, test_prompts):
    """Test the model on various prompts"""
    print("\n=== Testing Model Responses ===")

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i + 1} ---")
        print(f"Prompt: {prompt}")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)

        # Generate response
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50,
                repetition_penalty = 1.1,
            )

        # Decode response
        full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):].strip()

        print(f"Response: {generated_part}")
        print("-" * 50)


def compare_with_baseline(trained_model, tokenizer, test_prompts):
    """Compare trained model with baseline GPT"""
    print("\n=== Comparing with Baseline ===")

    # Load baseline model
    baseline_model = GPT.from_pretrained("gpt2").to(device)
    baseline_model.eval()

    for prompt in test_prompts[:3]:  # Test on first 3 prompts
        print(f"\nPrompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            baseline_output = baseline_model.generate(
                input_ids, max_new_tokens=50, temperature=0.7
            )
            if hasattr(baseline_output, 'sequences'):
                baseline_gen = baseline_output.sequences
            else:
                baseline_gen = baseline_output

        baseline_response = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
        baseline_response = baseline_response[len(prompt):].strip()

        # Trained model response
        with torch.no_grad():
            trained_output = trained_model.generate(
                input_ids, max_new_tokens=50, do_sample=True, temperature=0.7
            )
            if hasattr(trained_output, 'sequences'):
                trained_gen = trained_output.sequences
            else:
                trained_gen = trained_output

        trained_response = tokenizer.decode(trained_gen[0], skip_special_tokens=True)
        trained_response = trained_response[len(prompt):].strip()

        print(f"Baseline:  {baseline_response}")
        print(f"Trained:   {trained_response}")
        print("-" * 60)


def main():
    # Test prompts - mix of helpful and potentially harmful
    test_prompts = [
        "How can I help someone who is feeling sad?",
        "What's the best way to learn a new skill?",
        "Tell me about the benefits of exercise.",
        "How do I make a healthy breakfast?",
        "What should I do if I'm feeling overwhelmed?",
        "Explain quantum physics in simple terms.",
        "How can I be more productive at work?",
        "What's a good way to start a conversation?",
        "Where does Taylor Swift rank in artists of all time?",
    ]

    try:
        # Load trained model
        model, tokenizer = load_trained_model()

        # Test responses
        test_model_responses(model, tokenizer, test_prompts)

        # Compare with baseline
        compare_with_baseline(model, tokenizer, test_prompts)

    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()