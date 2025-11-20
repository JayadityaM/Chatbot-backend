import os
import json
import hashlib
from typing import List, Tuple, Dict
import numpy as np

class ChunkCache:
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, files: List[str], chunk_size: int, overlap: int) -> str:
        """Generate a cache key based on files and chunking parameters."""
        # Get modification times and sizes for all files
        file_info = []
        for file_path in sorted(files):
            stats = os.stat(file_path)
            file_info.append((file_path, stats.st_mtime, stats.st_size))
        
        # Create a string to hash
        hash_str = json.dumps({
            'files': file_info,
            'chunk_size': chunk_size,
            'overlap': overlap
        })
        
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def get_cached_chunks(self, 
                         files: List[str], 
                         chunk_size: int, 
                         overlap: int) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        """Try to load chunks and embeddings from cache."""
        cache_key = self._get_cache_key(files, chunk_size, overlap)
        chunks_path = os.path.join(self.cache_dir, f"{cache_key}_chunks.json")
        embeddings_path = os.path.join(self.cache_dir, f"{cache_key}_embeddings.npy")
        
        if os.path.exists(chunks_path) and os.path.exists(embeddings_path):
            # Load chunks and embeddings from cache
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_with_metadata = json.load(f)
            embeddings = np.load(embeddings_path)
            return chunks_with_metadata, embeddings
        
        return None, None
    
    def cache_chunks(self,
                    files: List[str],
                    chunk_size: int,
                    overlap: int,
                    chunks_with_metadata: List[Tuple[str, str]],
                    embeddings: np.ndarray) -> None:
        """Save chunks and embeddings to cache."""
        cache_key = self._get_cache_key(files, chunk_size, overlap)
        chunks_path = os.path.join(self.cache_dir, f"{cache_key}_chunks.json")
        embeddings_path = os.path.join(self.cache_dir, f"{cache_key}_embeddings.npy")
        
        # Save chunks to JSON
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)
        
        # Save embeddings to numpy file
        np.save(embeddings_path, embeddings)
