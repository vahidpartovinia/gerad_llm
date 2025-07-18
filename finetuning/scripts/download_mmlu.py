import kagglehub
import json
from pathlib import Path

def download_mmlu():
    """Download MMLU dataset using kagglehub"""
    print("Downloading MMLU dataset from Kaggle...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("peiyuanliu2001/mmlu-dataset")
        print(f"\nDataset downloaded to: {path}")
        
        # Verify the data
        data_dir = Path(path)
        splits = ['dev', 'val', 'test']
        
        for split in splits:
            file_path = data_dir / f"{split}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"\n{split.upper()} split:")
                    print(f"Number of questions: {len(data)}")
                    print("\nSample question:")
                    sample = data[0]
                    print(f"Question: {sample['question']}")
                    print("Choices:")
                    for i, choice in enumerate(sample['choices']):
                        print(f"{chr(65+i)}) {choice}")
                    print(f"Answer: {sample['answer']}")
            else:
                print(f"\nWarning: {split} split not found at {file_path}")
                
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_mmlu()
    print("\nDownload complete!") 