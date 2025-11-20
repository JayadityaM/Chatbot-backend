from typing import List, Tuple, Dict

class ContextManager:
    @staticmethod
    def prepare_context(chunks_with_scores: List[Tuple[str, str, float]], intent: str, similarity_threshold: float = 0.5) -> str:
        """Prepare and organize context for better answer generation."""
        # Filter and sort chunks
        filtered_chunks = []
        seen_content = set()
        
        for chunk, metadata, score in chunks_with_scores:
            if score < similarity_threshold:
                continue
                
            # Simple deduplication using first few words
            chunk_summary = ' '.join(chunk.lower().split()[:10])
            if chunk_summary in seen_content:
                continue
            
            seen_content.add(chunk_summary)
            filtered_chunks.append((chunk, metadata, score))
        
        # Sort by relevance score
        filtered_chunks.sort(key=lambda x: x[2], reverse=True)
        
        # Prepare context string
        context_parts = []
        for i, (chunk, metadata, score) in enumerate(filtered_chunks[:5], 1):
            relevance = "High" if score > 0.8 else "Medium" if score > 0.6 else "Low"
            context_parts.append(f"[Excerpt {i}]\n{chunk}\nSource: {metadata} | Relevance: {relevance}")
        
        return "\n\n---\n\n".join(context_parts)
    
    @staticmethod
    def generate_prompt(question: str, context: str, intent: str) -> str:
        """Generate an enhanced prompt for better answer extraction."""
        base_prompt = f"""Answer the following question using ONLY the information provided in the context below.
Be thorough and use ALL relevant information found in the context.

QUESTION: {question}

CONTEXT:
{context}

IMPORTANT INSTRUCTIONS:
1. If you find ANY information in the context that's even slightly relevant, include it in your answer
2. Look for both direct mentions and related information about the topic
3. If you find names or entities mentioned, include all associated details
4. Combine information from all available excerpts
5. If you find partial information, share it and indicate there might be more
6. Never say "information is not available" without thoroughly checking each excerpt
7. If uncertain about specific details, say "I'm not certain about [detail]" rather than saying no information exists
8. Pay attention to source relevance scores when weighing information

Additional Guidelines:"""

        # Add intent-specific guidelines
        if intent == "person":
            base_prompt += """
- Include any roles, titles, or positions mentioned
- Include any departments or affiliations
- Include any activities or responsibilities mentioned
- Include any contact information if available"""
        elif intent == "vacancy":
            base_prompt += """
- Include position details and requirements
- Include application deadlines if mentioned
- Include contact information for applications
- Note if the information might be time-sensitive"""
        elif intent == "location":
            base_prompt += """
- Include specific location details
- Include directions or landmarks if mentioned
- Include any associated facilities or services
- Include timing or accessibility information"""
        
        base_prompt += "\n\nBased on the context provided, give a detailed answer:"
        
        return base_prompt
