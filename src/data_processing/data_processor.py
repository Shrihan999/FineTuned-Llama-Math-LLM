from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple
import logging
import json
from pathlib import Path
from tqdm.auto import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from .dataset_loader import MetaMathDataLoader

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() 
            for key, val in self.encodings.items()
        }
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

class DataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Dict):
        self.tokenizer = tokenizer
        self.config = config
        self.data_loader = MetaMathDataLoader(config)
        self.processed_dir = Path(config['paths']['processed_data'])
        self.max_length = config['model']['max_length']
    
    def process_example(self, example: Dict) -> Dict:
        # Format input with instruction
        input_text = (
            f"You are a helpful math tutor. Solve this problem step by step:\n\n"
            f"Question: {example['query']}\n\nSolution:"
        )
        
        # Tokenize input and response using the new API
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize labels using text_target parameter
        labels = self.tokenizer(
            text_target=example['response'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            "input_ids": model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }
    
    def prepare_data(self) -> Tuple[CustomDataset, CustomDataset]:
        """
        Prepare dataset for training and validation
        Returns:
            Tuple of train and validation datasets
        """
        dataset = self.data_loader.load_dataset()
        train_data = dataset['train']
        
        # Process all examples
        processed_data = []
        for example in tqdm(train_data, desc="Processing examples"):
            processed_example = self.process_example(example)
            processed_data.append(processed_example)
        
        # Pre-allocate tensors
        total_size = len(processed_data)
        input_ids = torch.stack([ex['input_ids'] for ex in processed_data])
        attention_mask = torch.stack([ex['attention_mask'] for ex in processed_data])
        labels = torch.stack([ex['labels'] for ex in processed_data])
        
        # Calculate split sizes
        train_size = int(self.config['data']['train_split'] * total_size)
        
        # Create splits using tensor operations
        train_input_ids = input_ids[:train_size]
        train_attention_mask = attention_mask[:train_size]
        train_labels = labels[:train_size]
        
        val_input_ids = input_ids[train_size:]
        val_attention_mask = attention_mask[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets
        train_encodings = {
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask
        }
        val_encodings = {
            'input_ids': val_input_ids,
            'attention_mask': val_attention_mask
        }
        
        train_dataset = CustomDataset(train_encodings, train_labels)
        val_dataset = CustomDataset(val_encodings, val_labels)
        
        return train_dataset, val_dataset