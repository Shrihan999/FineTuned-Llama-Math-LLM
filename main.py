import argparse
import logging
import sys
from pathlib import Path
import yaml
import os
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

class Pipeline:
    def __init__(self):
        self.config = self.load_config()
        self.setup_directories()
        
    @staticmethod
    def load_config():
        """Load configuration from yaml file"""
        with open('config/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['paths']['raw_data'],
            self.config['paths']['processed_data'],
            self.config['paths']['model_output'],
            self.config['paths']['logs']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def process(self):
        """Process the MetaMathQA dataset"""
        from src.data_processing.data_processor import DataProcessor
        from src.model.model_loader import LlamaModelLoader
        
        logging.info("Starting data processing...")
        
        # Initialize components
        model_loader = LlamaModelLoader(self.config)
        _, tokenizer = model_loader.load_base_model_and_tokenizer()
        processor = DataProcessor(tokenizer, self.config)
        
        # Process data
        processor.process()
        logging.info("Data processing completed!")
    
    def train(self):
        """Train the model using QLoRA"""
        from src.model.trainer import ModelTrainer
        
        logging.info("Starting model training...")
        trainer = ModelTrainer(self.config)
        trainer.train()
        logging.info("Model training completed!")
    
    def start_ui(self):
        """Launch the Streamlit interface"""
        logging.info("Starting Streamlit interface...")
        streamlit_script = str(Path(__file__).parent / 'src' / 'ui' / 'app.py')
        subprocess.run(['streamlit', 'run', streamlit_script], check=True)

def main():
    parser = argparse.ArgumentParser(description="LLaMA Math Tuning Pipeline")
    parser.add_argument('--process', action='store_true', help='Process the dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--start', action='store_true', help='Start the Streamlit UI')
    
    args = parser.parse_args()
    
    pipeline = Pipeline()
    
    if args.process:
        pipeline.process()
    elif args.train:
        pipeline.train()
    elif args.start:
        pipeline.start_ui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()