import spacy
from transformers import pipeline
from typing import List, Dict
from integration_layer import IntegrationLayer
from feedback_loop import FeedbackLoop
from edge_computing import EdgeComputing
import logging
from sklearn.metrics.pairwise import cosine_similarity
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from leadership import King, Advisor, Noble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s"
)

class InputLayer:
    """
    The Input Layer processes raw user input, validates it, fetches external data,
    and generates realistic scenarios using LLMs. It serves as the entry point for
    user interactions and environmental data.
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.domain_models = {
            "general": pipeline("text-generation", model="gpt-4"),
            "healthcare": pipeline("text-generation", model="biobert"),
            "finance": pipeline("text-generation", model="finbert")
        }
        self.integration_layer = IntegrationLayer()
        self.feedback_loop = FeedbackLoop()
        self.edge_computing = EdgeComputing()
        self.keywords = {"optimize", "traffic", "reduce", "congestion", "smart", "city"}
        logging.info("InputLayer initialized successfully.")

    def preprocess(self, raw_input: str) -> List[str]:
        doc = self.nlp(raw_input)
        tokens = [token.text.lower() for token in doc if not token.is_punct]
        logging.info(f"Preprocessed tokens: {tokens}")
        return tokens

    def validate_input(self, tokens: List[str]) -> bool:
        if not tokens:
            logging.warning("Validation failed: Input is empty.")
            return False

        embedding = self._get_text_embedding(" ".join(tokens))
        valid_embeddings = [self._get_text_embedding(kw) for kw in self.keywords]
        similarities = [cosine_similarity([embedding], [ve])[0][0] for ve in valid_embeddings]
        if max(similarities) < 0.8:
            logging.warning("Validation failed: No relevant keywords or semantics found.")
            return False

        logging.info("Input validated successfully.")
        return True

    def fetch_external_data(self, api_url: str, retries: int = 3) -> Dict:
        for attempt in range(retries):
            try:
                data = self.integration_layer.fetch_external_data(api_url)
                if data:
                    logging.info(f"Fetched external data: {data}")
                    return data
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
        logging.error("All attempts failed. Returning empty data.")
        return {}

    def generate_scenario(self, prompt: str, domain: str = "general") -> str:
        generator = self.domain_models.get(domain, self.domain_models["general"])
        response = generator(prompt, max_length=50)
        scenario = response[0]['generated_text']
        logging.info(f"Generated scenario: {scenario}")
        return scenario

    def process_input(self, raw_input: str, api_url: str = None) -> Dict:
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

    def escalate_issue(self, issue: str, tier: str):
        if tier == "Noble":
            noble = Noble(region="North")
            noble.escalate_issue(issue)
        elif tier == "Advisor":
            advisor = Advisor(name="Advisor1")
            advisor.analyze_input(issue)
        elif tier == "King":
            king = King()
            king.set_vision(issue)
        logging.warning(f"Issue escalated to {tier}: {issue}")

    def _get_text_embedding(self, text: str):
        doc = self.nlp(text)
        return doc.vector

    def quantify_uncertainty(self: List[float]) -> float:
        model = BayesianNetwork([('A', 'B'), ('B', 'C')])
        cpd_a = TabularCPD('A', 2, [[0.6], [0.4]])
        model.add_cpds(cpd_a)
        uncertainty = model.get_cpds()[0].values[0]
        logging.info(f"Quantified uncertainty: {uncertainty:.4f}")
        return uncertainty
