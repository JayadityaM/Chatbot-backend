import spacy
from typing import Tuple, Dict
import re

class QueryProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.intent_patterns = {
            "person": [
                r"\b(?:who|professor|prof|dr|faculty|teacher|staff)\b",
                r"(?:who\s+is|about)\s+(?:prof|professor|dr|mr|mrs|ms)\s+\w+"
            ],
            "vacancy": [
                r"\b(?:vacancy|opening|position|job|recruitment|hire|hiring|apply)\b",
                r"(?:is|are)\s+there\s+(?:any|opening|vacancy|position)"
            ],
            "location": [
                r"\b(?:where|location|place|building|room|hall|lab|laboratory)\b",
                r"(?:where\s+(?:is|are|can|could|should|would|will))"
            ],
            "timing": [
                r"\b(?:when|time|schedule|deadline|date|timing|duration)\b",
                r"(?:what|which)\s+(?:time|date|day|month|year)"
            ],
            "document": [
                r"\b(?:document|pdf|file|form|certificate|application)\b"
            ]
        }
    
    def needs_clarification(self, query: str) -> bool:
        """Always return True as we want to clarify all queries."""
        return True
    
    def detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query."""
        query_lower = query.lower()
        
        # Check against each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return "general"
    
    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract relevant entities from the query."""
        doc = self.nlp(query)
        entities = {
            "names": [],
            "dates": [],
            "organizations": [],
            "locations": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["names"].append(ent.text)
            elif ent.label_ == "DATE":
                entities["dates"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE" or ent.label_ == "LOC":
                entities["locations"].append(ent.text)
                
        return {k: v for k, v in entities.items() if v}  # Remove empty lists
    
    def determine_search_scope(self, query: str, intent: str, entities: Dict) -> str:
        """Determine whether to search in specific files or globally."""
        # Queries that typically need broader context
        broad_intents = {"vacancy", "general"}
        # Queries that typically need specific context
        specific_intents = {"document", "timing"}
        
        if intent in broad_intents:
            return "global"
        elif intent in specific_intents:
            return "specific"
        
        # For person queries, check if they're mentioned in multiple contexts
        if intent == "person" and "names" in entities:
            return "global"  # Search broadly for person mentions
            
        # For location queries, check if they need broader context
        if intent == "location" and "locations" in entities:
            return "specific"  # Usually locations are in specific documents
            
        return "global"  # Default to global search for better coverage
    
    def generate_clarification(self, query: str) -> str:
        """Generate a detailed clarification based on query analysis."""
        doc = self.nlp(query)
        intent = self.detect_intent(query)
        entities = self.extract_entities(query)
        search_scope = self.determine_search_scope(query, intent, entities)
        
        # Build a comprehensive clarification
        clarification_parts = []
        
        # 1. Basic understanding confirmation
        if "names" in entities:
            clarification_parts.append(f"You're asking about {', '.join(entities['names'])}.")
        if "organizations" in entities:
            clarification_parts.append(f"This is regarding {', '.join(entities['organizations'])}.")
        if "locations" in entities:
            clarification_parts.append(f"This concerns the location {', '.join(entities['locations'])}.")
        if "dates" in entities:
            clarification_parts.append(f"For the time period: {', '.join(entities['dates'])}.")
        
        # 2. Intent-based clarification
        if intent == "person":
            clarification_parts.append("Would you like to know about their:\n- Role and responsibilities\n- Contact information\n- Publications or achievements?")
        elif intent == "vacancy":
            clarification_parts.append("Are you interested in:\n- Current openings\n- Application process\n- Eligibility criteria\n- Deadline information?")
        elif intent == "location":
            clarification_parts.append("Would you like to know:\n- Exact location\n- How to reach there\n- Working hours\n- Available facilities?")
        elif intent == "timing":
            clarification_parts.append("Are you looking for:\n- Specific dates\n- Deadlines\n- Schedule\n- Duration?")
        elif intent == "document":
            clarification_parts.append("Are you looking for:\n- Forms or applications\n- Guidelines\n- Certificates\n- Submission process?")
        
        # Combine all parts
        base_clarification = " ".join(clarification_parts)
        
        search_scope_msg = ""
        if search_scope == "global":
            search_scope_msg = "\n\nI'll search across all available documents for comprehensive information."
        else:
            search_scope_msg = "\n\nI'll focus on the most relevant documents for this query."
        
        return f"Let me clarify your query about '{query}':\n\n{base_clarification}{search_scope_msg}\n\nPlease confirm if this is what you're looking for."
        
    def process_query(self, query: str) -> Tuple[str, str, bool, str]:
        """
        Process the query and return intent, clarification message, whether clarification is needed, and search scope.
        Returns: (intent, clarification_message, needs_clarification, search_scope)
        """
        try:
            intent = self.detect_intent(query)
            entities = self.extract_entities(query)
            search_scope = self.determine_search_scope(query, intent, entities)
            needs_clarify = True  # Always ask for clarification
            
            clarification = self.generate_clarification(query)
            return intent, clarification, needs_clarify, search_scope
            
        except Exception as e:
            # If something goes wrong, return safe default values
            return "general", "Could you please provide more details about what you're looking for?", True, "global"
    
    def process_query(self, query: str) -> Tuple[str, str, bool]:
        """
        Process the query and return intent, clarification message, and whether clarification is needed.
        Returns: (intent, clarification_message, needs_clarification)
        """
        intent = self.detect_intent(query)
        needs_clarify = self.needs_clarification(query)
        
        if needs_clarify:
            clarification = self.generate_clarification(query)
            return intent, clarification, True
        
        return intent, "", False
