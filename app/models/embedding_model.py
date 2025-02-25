from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"  # A lightweight embedding model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for a text or list of texts
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

# Create a singleton instance
embedding_model = EmbeddingModel()