import chromadb
from chromadb.config import Settings
import logging
import os
from typing import List, Dict, Any, Optional
from app.models.embedding_model import embedding_model

logger = logging.getLogger(__name__)

class ChromaDBHandler:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "document_store"
        self.db_directory = "data/chroma_db"
        self.initialize_db()

    def initialize_db(self):
        """Initialize the ChromaDB client and collection"""
        try:
            # Ensure the directory exists
            os.makedirs(self.db_directory, exist_ok=True)
            
            # Create a persistent client
            self.client = chromadb.PersistentClient(path=self.db_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized. Collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None):
        """
        Add documents to the vector database
        
        Args:
            texts: List of text chunks to add
            metadatas: List of metadata dictionaries for each chunk
            ids: Optional list of IDs (if None, will be auto-generated)
        """
        try:
            # Generate embeddings
            embeddings = embedding_model.get_embeddings(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = embedding_model.get_embeddings(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise

# Create a singleton instance
chroma_db = ChromaDBHandler()