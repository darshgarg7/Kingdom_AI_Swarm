import logging
from typing import Dict
from feedback import FeedbackLoop
from edge_computing import EdgeComputing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s"
)

class IntegrationLayer:
    """
    Simulates an external data source (can be replaced with real APIs).
    """
    def __init__(self):
        """
        Initializes the IntegrationLayer with mock data for testing.
        """
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

    def fetch_external_data(self, api_url: str) -> Dict:
        """
        Fetches data from an external API (simulated with mock data).
        Args:
            api_url (str): URL of the external API.
        Returns:
            Dict: External data fetched from the API.
        """
        if api_url in self.mock_data:
            logging.info(f"Data fetched successfully from {api_url}")
            return self.mock_data[api_url]
        else:
            logging.error(f"Failed to fetch data from {api_url}")
            return {}

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.generator = pipeline("text-generation", model="gpt-4")
        self.integration_layer = IntegrationLayer()
        self.feedback_loop = FeedbackLoop()
        self.edge_computing = EdgeComputing()
        self.keywords = {"optimize", "traffic", "reduce", "congestion", "smart", "city"}
        logging.info("InputLayer initialized successfully.")

    def process_input(self, raw_input: str, api_url: str = None) -> Dict:
        """
        Processes user input dynamically:
        1. Preprocesses the input.
        2. Validates the input.
        3. Fetches external data if an API URL is provided.
        4. Generates a scenario if needed.
        5. Updates the FeedbackLoop and EdgeComputing components.
        Args:
            raw_input (str): Raw user input.
            api_url (str, optional): URL of the external API.
        Returns:
            Dict: Processed input data.
        """
        logging.info(f"Raw input received: {raw_input}")

        tokens = self.preprocess(raw_input)

        is_valid = self.validate_input(tokens)
        if not is_valid:
            return {"status": "failed", "message": "Input validation failed."}

        external_data = {}
        if api_url:
            external_data = self.fetch_external_data(api_url)

        scenario = self.generate_scenario(raw_input)

        self.feedback_loop.collect_feedback(raw_input, tier="Worker")
        self.edge_computing.process_locally({"input": raw_input})

        return {
            "status": "success",
            "tokens": tokens,
            "external_data": external_data,
            "scenario": scenario
        }
    