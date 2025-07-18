 import torch
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

from model import GPT
from config import nano_config
from data import ShakespeareDataset
from model_config import get_model_config

def load_model(model_name, device):
    config = get_model_config(model_name)
    
    if config.model_type == "nano":
        # Load nano model
        model_path = "nano_gpt_mmlu_best.pt"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the Shakespeare dataset to get the tokenizer
        shakespeare_dataset = ShakespeareDataset("data", nano_config.block_size)
        nano_config.vocab_size = shakespeare_dataset.vocab_size
        
        # Create and load model
        model = GPT(nano_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        return model, shakespeare_dataset
        
    elif config.model_type == "gpt2":
        # Load GPT-2 model
        model_path = Path("models/gpt2")
        if not model_path.exists():
            raise FileNotFoundError("GPT-2 model not found. Please run download_gpt2.py first.")
        
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = model.to(device)
        model.eval()
        return model, tokenizer

def format_question(prompt, choices, model_type):
    if model_type == "nano":
        text = f"Question: {prompt}\n"
        for i, choice in enumerate(choices):
            text += f"{chr(65+i)}) {choice}\n"
        return text
    else:  # gpt2
        text = f"Question: {prompt}\n"
        for i, choice in enumerate(choices):
            text += f"{chr(65+i)}) {choice}\n"
        text += "Answer: "  # GPT-2 will complete this
        return text

def get_model_prediction(model, tokenizer, question_text, device, model_type):
    if model_type == "nano":
        # Convert question to tokens
        tokens = [tokenizer.stoi.get(c, 0) for c in question_text]
        if len(tokens) > nano_config.block_size:
            tokens = tokens[:nano_config.block_size]
        
        # Convert to tensor and add batch dimension
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            logits, _ = model(x)
            # Get the last token's logits
            last_token_logits = logits[0, -1, :]
            # Get probabilities for A, B, C, D
            probs = torch.softmax(last_token_logits[:4], dim=0)
            # Get the predicted answer
            pred_idx = torch.argmax(probs).item()
            pred_answer = chr(65 + pred_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        return pred_answer, probs.cpu().numpy()
    
    else:  # gpt2
        # Tokenize input
        inputs = tokenizer(question_text, return_tensors="pt").to(device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]
            # Get probabilities for A, B, C, D tokens
            a_token_id = tokenizer.encode("A")[0]
            probs = torch.softmax(next_token_logits[a_token_id:a_token_id+4], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_answer = chr(65 + pred_idx)
        
        return pred_answer, probs.cpu().numpy()

def evaluate_model(model, tokenizer, test_file, device, model_type, num_samples=5):
    # Load test data
    test_data = pd.read_csv(test_file)
    
    # Initialize counters
    correct = 0
    total = 0
    
    # Store some sample predictions
    samples = []
    
    # Evaluate
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating"):
        # Format question
        question_text = format_question(row['prompt'], [row['A'], row['B'], row['C'], row['D']], model_type)
        
        # Get model prediction
        pred_answer, probs = get_model_prediction(model, tokenizer, question_text, device, model_type)
        
        # Check if correct
        is_correct = pred_answer == row['answer']
        correct += int(is_correct)
        total += 1
        
        # Store sample if it's one of the first num_samples
        if idx < num_samples:
            samples.append({
                'question': row['prompt'],
                'choices': [row['A'], row['B'], row['C'], row['D']],
                'correct_answer': row['answer'],
                'predicted_answer': pred_answer,
                'probabilities': {chr(65+i): float(p) for i, p in enumerate(probs)},
                'is_correct': is_correct
            })
    
    # Calculate accuracy
    accuracy = correct / total
    
    return accuracy, samples

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on MMLU dataset')
    parser.add_argument('--model', type=str, default='nano', choices=['nano', 'gpt2'],
                      help='Model to evaluate (nano or gpt2)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"Loading {args.model} model...")
    model, tokenizer = load_model(args.model, device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy, samples = evaluate_model(model, tokenizer, "mmlu_dataset/test.csv", device, args.model)
    
    # Print results
    print(f"\nTest Accuracy: {accuracy:.2%}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample['question']}")
        print("Choices:")
        for j, choice in enumerate(sample['choices']):
            print(f"{chr(65+j)}) {choice}")
        print(f"Correct Answer: {sample['correct_answer']}")
        print(f"Predicted Answer: {sample['predicted_answer']}")
        print("Answer Probabilities:")
        for ans, prob in sample['probabilities'].items():
            print(f"{ans}: {prob:.2%}")
        print(f"Correct: {'✓' if sample['is_correct'] else '✗'}")

if __name__ == "__main__":
    main() 