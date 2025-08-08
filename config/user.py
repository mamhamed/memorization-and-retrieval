from dataclasses import dataclass


@dataclass
class UserProfile:
    """User profile data structure"""
    id: int
    name: str
    city: str
    education: str
    website: str
    work: str
    relationship: str
    language: str
