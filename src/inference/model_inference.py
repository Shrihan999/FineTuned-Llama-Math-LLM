from typing import Dict, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from ..model.model_loader import LlamaModelLoader

class LlamaInference:
    def __init__(self, config: Dict):
        """
        Initialize the inference class
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_loader = LlamaModelLoader(config)
        self.base_model = None
        self.finetuned_model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_models(self):
        """Load both base and fine-tuned models"""
        logging.info("Loading models for inference...")
        
        # Load base model and tokenizer
        self.base_model, self.tokenizer = self.model_loader.load_base_model_and_tokenizer()
        
        # Load fine-tuned model
        try:
            self.finetuned_model = self.model_loader.load_finetuned_model()
        except ValueError as e:
            logging.warning(f"Could not load fine-tuned model: {e}")
            self.finetuned_model = None
        
        logging.info("Models loaded successfully")
    
    def generate_response(
        self,
        question: str,
        use_finetuned: bool = True,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict:
        """
        Generate response for a given question
        Args:
            question: Input question
            use_finetuned: Whether to use fine-tuned model
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
        Returns:
            Dictionary containing response and metadata
        """
        # Load models if not already loaded
        if self.base_model is None or self.tokenizer is None:
            self.load_models()
        
        # Select model
        model = self.finetuned_model if use_finetuned and self.finetuned_model else self.base_model
        
        # Format input prompt
        prompt = (
            f"You are a helpful math tutor. Solve this problem step by step:\n\n"
            f"Question: {question}\n\nSolution:"
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['model']['max_length']
        ).to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature or self.config['model']['temperature'],
            "top_p": top_p or self.config['model']['top_p'],
            "num_beams": self.config['model']['num_beams'],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        
        # Generate response
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(
            output_sequences[0],
            skip_special_tokens=True
        )
        
        # Extract just the solution part
        solution = response.split("Solution:")[-1].strip()
        
        return {
            "question": question,
            "solution": solution,
            "model_type": "fine-tuned" if use_finetuned else "base",
            "generation_params": gen_kwargs
        }
    
    def batch_evaluate(
        self,
        questions: List[str],
        use_finetuned: bool = True
    ) -> List[Dict]:
        """
        Evaluate model on a batch of questions
        Args:
            questions: List of questions to evaluate
            use_finetuned: Whether to use fine-tuned model
        Returns:
            List of response dictionaries
        """
        responses = []
        for question in questions:
            response = self.generate_response(question, use_finetuned)
            responses.append(response)
        return responses
    
    def compare_models(self, question: str) -> Dict:
        """
        Compare responses from both base and fine-tuned models
        Args:
            question: Input question
        Returns:
            Dictionary containing both responses
        """
        base_response = self.generate_response(question, use_finetuned=False)
        finetuned_response = self.generate_response(question, use_finetuned=True)
        
        return {
            "question": question,
            "base_model_response": base_response["solution"],
            "finetuned_model_response": finetuned_response["solution"]
        }