from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QwenModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load the Qwen model and tokenizer"""
        try:
            logger.info(f"Loading Qwen model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            logger.info("Qwen model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_response(self, prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the Qwen model
        
        Args:
            prompt (str): The user input
            history (List[Dict]): Chat history in format [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            str: The model's response
        """
        try:
            if history is None:
                history = []
                
            # Format the conversation history for Qwen
            formatted_history = []
            for message in history:
                formatted_history.append({"role": message["role"], "content": message["content"]})
                
            # Add the current prompt
            formatted_history.append({"role": "user", "content": prompt})
            
            input_text = self.tokenizer.apply_chat_template(formatted_history, tokenize=False)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.7)

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def generate_response_with_context(self, prompt: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the Qwen model with additional context from documents or database
        
        Args:
            prompt (str): The user input
            context (str): Additional context from documents or database query results
            history (List[Dict]): Chat history
            
        Returns:
            str: The model's response
        """
        try:
            if history is None:
                history = []
                
            # Create a prompt that includes the context
            enhanced_prompt = f"""Please use the following information to answer the user's question.
            
Context information:
{context}

User question: {prompt}

Please answer based on the context information provided. If the context doesn't contain relevant information, say so."""

            # Generate response with the enhanced prompt
            return self.generate_response(enhanced_prompt, history)
        except Exception as e:
            logger.error(f"Error generating response with context: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

# Create a singleton instance
qwen_model = QwenModel()