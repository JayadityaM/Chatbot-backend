import re
from typing import List, Tuple

def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text

# Constants for chunking
CHUNK_SIZE = 1000
OVERLAP = 200

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    # Preprocess the text first
    text = preprocess_text(text)
    
    # Split into sentences (rough approximation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        if current_length + sentence_length > chunk_size:
            # Join the current chunk and add it to chunks
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if chunks:
                # Find sentences from previous chunk for overlap
                overlap_text = ''
                overlap_length = 0
                for prev_sentence in reversed(current_chunk):
                    if overlap_length + len(prev_sentence) > overlap:
                        break
                    overlap_text = prev_sentence + ' ' + overlap_text
                    overlap_length += len(prev_sentence)
                
                current_chunk = overlap_text.strip().split()
                current_length = overlap_length
            else:
                current_chunk = []
                current_length = 0
                
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_document(file_path: str) -> List[Tuple[str, str]]:
    """Process a document and return chunks with metadata."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = create_chunks(text)
    # Add metadata to each chunk
    doc_chunks = [(chunk, f"Source: {file_path}, Position: {i+1}/{len(chunks)}") 
                 for i, chunk in enumerate(chunks)]
    
    return doc_chunks
