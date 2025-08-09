from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from typing import Dict, Tuple, Optional
import torch
import logging
from pathlib import Path

class LlamaModelLoader:
    def __init__(self, config: Dict):
        """
        Initialize the Llama model loader
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_path = Path(config['paths']['model_output'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_base_model_and_tokenizer(
        self
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the base Llama model and tokenizer with quantization config
        Returns:
            Tuple of model and tokenizer
        """
        logging.info("Loading base model and tokenizer...")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['name'],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logging.info("Base model and tokenizer loaded successfully")
        return model, tokenizer
    
    def load_finetuned_model(
        self,
        checkpoint_path: Optional[str] = None
    ) -> AutoModelForCausalLM:
        """
        Load the fine-tuned model
        Args:
            checkpoint_path: Optional specific checkpoint path
        Returns:
            Fine-tuned model
        """
        if checkpoint_path is None:
            # Use latest checkpoint if not specified
            checkpoints = list(self.model_path.glob("checkpoint-*"))
            if not checkpoints:
                raise ValueError("No checkpoints found in model output directory")
            checkpoint_path = str(sorted(checkpoints)[-1])
        
        logging.info(f"Loading fine-tuned model from {checkpoint_path}")
        
        # Load with same quantization config as base model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        logging.info("Fine-tuned model loaded successfully")
        return model