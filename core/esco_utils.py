from pydantic import BaseModel
from typing import List, Optional

class ESCONode(BaseModel):
    esco_code: str
    preferred_label: str
    parent_code: Optional[str] = None
    
def verify_esco_node(esco_code: str) -> bool:
    """
    Mock ESCO API validation.
    Returns True if the node is valid, False otherwise (simulating a 404).
    """
    # In a real scenario, this would query a database or API.
    # For now, let's just pretend anything starting with "ESCO" or valid digits is good.
    if not esco_code:
        return False
    return True

def fallback_vector_search(raw_skill: str) -> ESCONode:
    """
    If the ESCO API returns 404, we use vector similarity to find the closest ESCO node.
    """
    # Mocking a fallback search
    return ESCONode(esco_code="0000.0.0", preferred_label=raw_skill, parent_code=None)
