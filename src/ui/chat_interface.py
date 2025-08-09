import streamlit as st
from typing import Dict
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.inference.model_inference import LlamaInference

class ChatInterface:
    def __init__(self, config: Dict):
        """
        Initialize the chat interface
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.inferencer = LlamaInference(config)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'model_choice' not in st.session_state:
            st.session_state.model_choice = 'Compare Both'
            
    def display_message(self, role: str, content: str, model_type: str = None):
        """Display a single message in the chat"""
        with st.chat_message(role):
            if model_type:
                st.markdown(f"**{model_type} Response:**")
            st.markdown(content)
            
    def display_chat_history(self):
        """Display the chat history"""
        for message in st.session_state.messages:
            if message["role"] == "user":
                self.display_message("user", message["content"])
            else:
                if message["model_type"] == "Compare Both":
                    self.display_message(
                        "assistant", 
                        message["base_content"],
                        "Base Model"
                    )
                    self.display_message(
                        "assistant", 
                        message["ft_content"],
                        "Fine-tuned Model"
                    )
                else:
                    self.display_message(
                        "assistant",
                        message["content"],
                        message["model_type"]
                    )
                    
    def get_model_response(self, question: str) -> Dict:
        """
        Get response based on selected model
        Args:
            question: User's question
        Returns:
            Dictionary containing response(s)
        """
        if st.session_state.model_choice == 'Compare Both':
            comparison = self.inferencer.compare_models(question)
            return {
                "role": "assistant",
                "model_type": "Compare Both",
                "base_content": comparison["base_model_response"],
                "ft_content": comparison["finetuned_model_response"]
            }
        else:
            use_finetuned = st.session_state.model_choice == 'Fine-tuned Model'
            response = self.inferencer.generate_response(
                question,
                use_finetuned=use_finetuned
            )
            return {
                "role": "assistant",
                "model_type": st.session_state.model_choice,
                "content": response["solution"]
            }
            
    def render(self):
        """Render the chat interface"""
        st.title("Math Problem Solver")
        st.markdown("""
        Ask me any math question and I'll solve it step by step.
        You can compare responses from the base and fine-tuned models
        or use them individually.
        """)
        
        # Initialize session state
        self.initialize_session_state()
        
        # Display chat history
        self.display_chat_history()
        
        # Chat input
        if question := st.chat_input("Enter your math question..."):
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            # Get model response
            with st.spinner("Thinking..."):
                response = self.get_model_response(question)
                
            # Add response to history
            st.session_state.messages.append(response)
            
            # Rerun to update display
            st.rerun()

    def clear_chat(self):
        """Clear the chat history"""
        st.session_state.messages = []