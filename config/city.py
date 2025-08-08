from dataclasses import dataclass


@dataclass
class CityInfo:
    """City information data structure"""
    city: str
    mayor: str
    population: int
    temperature: float
    timezone: str
    description: str
