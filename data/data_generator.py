from typing import List, Dict
import random
import numpy as np
from config.user import UserProfile
from config.city import CityInfo
from faker import Faker
from jinja2 import Template
from dataclasses import dataclass, asdict

class DataGenerator:
    """Handles synthetic data generation"""
    
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Templates for CPT data diversification
        self.profile_templates = [
            # Template 1: Narrative
            "{{name}} lives in {{city}}, works as {{work}} after studying at {{education}}. They are currently {{relationship}} and share more information on {{website}}.",
            
            # Template 2: YAML Style
            """------------------
name: {{name}}
city: {{city}}
work: {{work}}
education: {{education}}
relationship: {{relationship}}
website: {{website}}
------------------""",
            
            # Template 3: JSON
            """{
  "name": "{{name}}",
  "city": "{{city}}",
  "work": "{{work}}",
  "education": "{{education}}",
  "relationship": "{{relationship}}",
  "website": "{{website}}"
}""",
            
            # Template 4: Bullet Points
            """• Name: {{name}}
• Location: {{city}}
• Occupation: {{work}}
• Education: {{education}}
• Status: {{relationship}}
• Website: {{website}}""",
            
            # Template 5: HTML Style
            """<profile>
  <name>{{name}}</name>
  <city>{{city}}</city>
  <work>{{work}}</work>
  <education>{{education}}</education>
  <relationship>{{relationship}}</relationship>
  <website>{{website}}</website>
</profile>""",
            
            # Template 6: Table Format
            """Profile Information:
Name        | {{name}}
City        | {{city}}
Work        | {{work}}
Education   | {{education}}
Status      | {{relationship}}
Website     | {{website}}""",
            
            # Template 7: Key-Value
            """name={{name}};city={{city}};work={{work}};education={{education}};relationship={{relationship}};website={{website}}""",
            
            # Template 8: Markdown
            """# {{name}}
**City:** {{city}}
**Work:** {{work}}
**Education:** {{education}}
**Relationship:** {{relationship}}
**Website:** {{website}}""",
            
            # Template 9: CSV Style
            """{{name}},{{city}},{{work}},{{education}},{{relationship}},{{website}}""",
            
            # Template 10: Paragraph
            """This is {{name}}'s profile. {{name}} currently resides in {{city}} and is employed in {{work}}. Educational background includes {{education}}. Current relationship status is {{relationship}}. More details at {{website}}."""
        ]
        
        self.city_templates = [
            # Template variations for city descriptions
            "{{city}} is a vibrant city with {{population}} residents. The current mayor is {{mayor}}. The average temperature is {{temperature}}°C and it's in the {{timezone}} timezone.",
            "Located in the {{timezone}} timezone, {{city}} has a population of {{population}}. {{mayor}} serves as mayor. The city enjoys an average temperature of {{temperature}}°C.",
            "With {{mayor}} as mayor, {{city}} is home to {{population}} people. The climate averages {{temperature}}°C in the {{timezone}} timezone.",
            "{{city}} ({{timezone}}): Population {{population}}, Mayor {{mayor}}, Average temp {{temperature}}°C",
            "The city of {{city}} has {{population}} inhabitants and is governed by Mayor {{mayor}}. Temperature averages {{temperature}}°C in {{timezone}}."
        ]
    
    def generate_profiles(self, n: int) -> List[UserProfile]:
        """Generate synthetic user profiles"""
        profiles = []
        cities = [self.fake.city() for _ in range(min(n//10, 1000))]  # Ensure city reuse
        
        for i in range(n):
            profile = UserProfile(
                id=i + 1,
                name=self.fake.name(),
                city=random.choice(cities),
                education=f"{self.fake.catch_phrase()} University",
                website=self.fake.url(),
                work=self.fake.job(),
                relationship=random.choice(['Single', 'Married', 'In a relationship']),
                language=random.choice(['English', 'Spanish', 'French', 'German'])
            )
            profiles.append(profile)
        
        return profiles
    
    def generate_city_info(self, cities: List[str]) -> List[CityInfo]:
        """Generate city information"""
        city_info = []
        
        for city in set(cities):
            info = CityInfo(
                city=city,
                mayor=self.fake.name(),
                population=random.randint(10000, 1000000),
                temperature=round(random.uniform(5, 35), 1),
                timezone=random.choice(['EST', 'PST', 'MST', 'CST', 'GMT']),
                description=""
            )
            # Generate description using template
            template = Template(random.choice(self.city_templates))
            info.description = template.render(asdict(info))
            city_info.append(info)
        
        return city_info
    
    def diversify_cpt_data(self, profiles: List[UserProfile], cities: List[CityInfo], 
                          num_variations: int = 10) -> List[str]:
        """Create diversified CPT training data"""
        cpt_data = []
        
        print(f"number of profiles in cpt: {len(profiles)}")
        print(f"number of cities in cpt: {len(cities)}")
        # Diversify profile data
        for profile in profiles:
            for i in range(min(num_variations, len(self.profile_templates))):
                template = Template(self.profile_templates[i])
                text = template.render(asdict(profile))
                cpt_data.append(text)
        
        # Diversify city data
        for city in cities:
            for i in range(min(5, len(self.city_templates))):  # Use 5 variations for cities
                template = Template(self.city_templates[i])
                text = template.render(asdict(city))
                cpt_data.append(text)
        
        return cpt_data
    
    def generate_qa_pairs(self, profiles: List[UserProfile], cities: List[CityInfo],
                         include_cot: bool = False) -> List[Dict[str, str]]:
        """Generate question-answer pairs for IFT"""
        qa_pairs = []
        
        # Create city lookup dict
        city_dict = {city.city: city for city in cities}
        
        # Profile questions
        for profile in profiles:
            questions = [
                (f"What is the name of user {profile.id}?", profile.name),
                (f"Where does user {profile.id} live?", profile.city),
                (f"What is user {profile.id}'s job?", profile.work),
                (f"What is user {profile.id}'s education?", profile.education),
                (f"What is user {profile.id}'s relationship status?", profile.relationship),
                (f"What is user {profile.id}'s website?", profile.website),
                (f"What language does user {profile.id} speak?", profile.language),
            ]
            
            for q, a in questions:
                qa_pairs.append({"question": q, "answer": a, "type": "profile"})
        
        # City questions
        for city in cities:
            questions = [
                (f"Who is the mayor of {city.city}?", city.mayor),
                (f"What is the population of {city.city}?", str(city.population)),
                (f"What is the average temperature in {city.city}?", f"{city.temperature}°C"),
                (f"What timezone is {city.city} in?", city.timezone),
            ]
            
            for q, a in questions:
                qa_pairs.append({"question": q, "answer": a, "type": "city"})
        
        # Two-hop questions
        for profile in profiles:
            if profile.city in city_dict:
                city_info = city_dict[profile.city]
                
                # Question about user's city's mayor
                question = f"Who is the mayor of the city where user {profile.id} lives?"
                if include_cot:
                    cot_answer = f"First, let me find where user {profile.id} lives. User {profile.id} lives in {profile.city}. Now, let me find who is the mayor of {profile.city}. The mayor of {profile.city} is {city_info.mayor}."
                    qa_pairs.append({
                        "question": question, 
                        "answer": cot_answer, 
                        "type": "two_hop_cot"
                    })
                else:
                    qa_pairs.append({
                        "question": question, 
                        "answer": city_info.mayor, 
                        "type": "two_hop"
                    })
                
                # Question about user's city's population
                question = f"What is the population of the city where user {profile.id} lives?"
                if include_cot:
                    cot_answer = f"First, let me find where user {profile.id} lives. User {profile.id} lives in {profile.city}. Now, let me find the population of {profile.city}. The population of {profile.city} is {city_info.population}."
                    qa_pairs.append({
                        "question": question, 
                        "answer": cot_answer, 
                        "type": "two_hop_cot"
                    })
                else:
                    qa_pairs.append({
                        "question": question, 
                        "answer": str(city_info.population), 
                        "type": "two_hop"
                    })
        
        return qa_pairs

