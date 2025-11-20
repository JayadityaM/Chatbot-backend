import numpy as np
import faiss
from typing import List, Tuple
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SearchEngine:
    def __init__(self, embedding_dim: int):
        # Initialize FAISS index with cosine similarity
        self.dimension = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        self.chunks = []
        self.metadata = []
        
    def add_documents(self, chunks: List[Tuple[str, str]], embeddings: np.ndarray):
        # Normalize vectors for cosine similarity
        normalized_vectors = embeddings.copy()
        faiss.normalize_L2(normalized_vectors)
        
        self.index.add(normalized_vectors)
        self.chunks = [chunk for chunk, _ in chunks]
        self.metadata = [meta for _, meta in chunks]
    
    def _process_query(self, query: str) -> str:
        """Process the query to improve search relevance."""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters
        query = re.sub(r'[^\w\s?]', '', query)
        
        # Tokenize
        tokens = word_tokenize(query)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        
        # Extract key terms (e.g., numbers, proper nouns)
        key_terms = []
        for token in tokens:
            if token.isdigit() or token[0].isupper():
                key_terms.append(token)
        
        # Combine processed tokens with original query
        processed_query = " ".join(tokens)
        if key_terms:
            processed_query = f"{processed_query} {' '.join(key_terms)}"
            
        return processed_query
    
    def _rerank_results(self, query: str, chunks: List[str], scores: np.ndarray) -> List[Tuple[int, float]]:
        """Rerank results based on additional criteria."""
        reranked = []
        
        for idx, (chunk, score) in enumerate(zip(chunks, scores)):
            # Base score from vector similarity
            final_score = score
            
            # Boost exact matches
            if query.lower() in chunk.lower():
                final_score *= 1.2
                
            # Boost shorter chunks that contain the answer
            length_penalty = 1.0 / (1.0 + np.log(len(chunk) / 100))
            final_score *= length_penalty
            
            # Boost chunks with question-relevant terms
            question_words = {'what', 'when', 'where', 'who', 'why', 'how'}
            query_words = set(query.lower().split())
            if any(word in query_words for word in question_words):
                relevant_terms = {
                    'what': ['is', 'are', 'was', 'were'],
                    'when': ['date', 'time', 'year', 'month', 'day'],
                    'where': ['location', 'place', 'at', 'in'],
                    'who': ['person', 'people', 'name'],
                    'why': ['because', 'reason', 'due to'],
                    'how': ['method', 'way', 'process']
                }
                for q_word in query_words & question_words:
                    if any(term in chunk.lower() for term in relevant_terms.get(q_word, [])):
                        final_score *= 1.1
            
            reranked.append((idx, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked
    
    def search(self, query_embedding: np.ndarray, query_text: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Search for relevant chunks using enhanced similarity search and reranking."""
        # Normalize query vector for cosine similarity
        query_vector = query_embedding.copy()
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        # Initial search
        scores, indices = self.index.search(query_vector.reshape(1, -1), top_k * 2)  # Get more results for reranking
        
        # Convert to lists
        retrieved_chunks = [self.chunks[idx] for idx in indices[0]]
        retrieved_metadata = [self.metadata[idx] for idx in indices[0]]
        
        # Rerank results
        reranked = self._rerank_results(query_text, retrieved_chunks, scores[0])
        
        # Return top k after reranking
        results = []
        for idx, score in reranked[:top_k]:
            results.append((
                retrieved_chunks[idx],
                retrieved_metadata[idx],
                float(score)
            ))
        
        return results
