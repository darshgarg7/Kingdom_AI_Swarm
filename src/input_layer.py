import requests
import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

class InputLayer:
    def __init__(self):
        self.generator = pipeline("text-generation", model="gpt-4")

    def generate_scenario(self, prompt: str):
        """
        Generates a realistic scenario using GPT-4.
        """
        response = self.generator(prompt, max_length=50)
        scenario = response[0]['generated_text']
        print(f"Generated scenario: {scenario}")
        return scenario

    def preprocess(self, raw_input: str) -> list:
        """
        Preprocesses raw user input using spaCy.
        """
        doc = nlp(raw_input)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        return tokens

    def validate_input(self, tokens: list) -> bool:
        """
        Validates preprocessed input to ensure it meets requirements.
        """
        if not tokens:
            print("Validation failed: Input is empty.")
            return False

        keywords = {"optimize", "traffic", "reduce", "congestion"}
        if not any(token in keywords for token in tokens):
            print("Validation failed: No relevant keywords found.")
            return False

        print("Input validated successfully.")
        return True

class IntegrationLayer:
    def __init__(self):
        # Simulated external data source (can be replaced with real data)
        self.mock_data = {
            "https://api.example.com/traffic": {
                "region": "North",
                "congestion_level": 0.75,
                "incident_reports": ["accident", "roadblock"]
            },
            "https://api.example.com/weather": {
                "region": "South",
                "temperature": 25,
                "weather_condition": "sunny"
            }
        }

    def fetch_external_data(self, api_url: str):
        """
        Fetches data from an external API (simulated with mock data).
        """
        if api_url in self.mock_data:
            print(f"Data fetched successfully from {api_url}")
            return self.mock_data[api_url]
        else:
            print(f"Failed to fetch data from {api_url}")
            return None
    