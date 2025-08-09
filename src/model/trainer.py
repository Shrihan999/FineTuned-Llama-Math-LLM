from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import torch
from typing import Dict
import logging
from pathlib import Path
from .model_loader import LlamaModelLoader
from ..data_processing.data_processor import DataProcessor

class ModelTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model_loader = LlamaModelLoader(config)
        self.output_dir = Path(config['paths']['model_output'])
        
    def prepare_model_for_training(self):
        # Load base model and tokenizer
        model, tokenizer = self.model_loader.load_base_model_and_tokenizer()
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['qlora']['r'],
            lora_alpha=self.config['qlora']['lora_alpha'],
            target_modules=self.config['qlora']['target_modules'],
            lora_dropout=self.config['qlora']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        
        return model, tokenizer
    
    def train(self):
        logging.info("Preparing for training...")
        
        # Prepare model and tokenizer
        model, tokenizer = self.prepare_model_for_training()
        
        # Load and process data
        data_processor = DataProcessor(tokenizer, self.config)
        train_dataset, eval_dataset = data_processor.prepare_data()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            warmup_ratio=self.config['training']['warmup_ratio'],
            max_steps=self.config['training']['max_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['training']['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['training']['save_steps'],
            logging_steps=self.config['training']['logging_steps'],
            weight_decay=self.config['training']['weight_decay'],
            fp16=True,
            optim="paged_adamw_32bit",
            logging_dir=str(Path(self.config['paths']['logs']) / "training"),
            load_best_model_at_end=True,
            report_to="tensorboard"
        )
        
        # Use proper data collator for seq2seq tasks
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model(str(self.output_dir / "final"))
        logging.info("Training completed successfully!")