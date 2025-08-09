import streamlit as st
import yaml
from pathlib import Path
import sys

# Add the project root to Python path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.ui.chat_interface import ChatInterface

def load_config() -> dict:
    """Load configuration from yaml file"""
    config_path = Path(__file__).parents[2] / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Page config
    st.set_page_config(
        page_title="Math Problem Solver",
        page_icon="🔢",
        layout="wide"
    )
    
    # Load configuration
    config = load_config()
    
    # Initialize chat interface
    chat = ChatInterface(config)
    
    # Sidebar
    with st.sidebar:
        st.title("🔢 Math Problem Solver")
        st.markdown("""
        This application uses two LLaMA models to solve math problems:
        1. Base Model: Original LLaMA 3.2 3B
        2. Fine-tuned Model: Specialized for math problems
        """)
        
        # Add clear chat button
        if st.button("Clear Chat"):
            chat.clear_chat()
            st.rerun()
        
        # Add some example questions
        st.markdown("### Example Questions")
        examples = [
            "Solve for x: 2x + 5 = 13",
            "Find the derivative of f(x) = x³ + 2x² - 4x + 1",
            "What is the area of a circle with radius 5?",
            "Simplify: (x² + 2x + 1) - (x² - 2x + 4)"
        ]
        
        st.markdown("Try these examples:")
        for example in examples:
            if st.button(example):
                # Add example to chat
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({
                    "role": "user",
                    "content": example
                })
                # Get response
                with st.spinner("Thinking..."):
                    response = chat.get_model_response(example)
                st.session_state.messages.append(response)
                st.rerun()
    
    # Render chat interface
    chat.render()

if __name__ == "__main__":
    main()