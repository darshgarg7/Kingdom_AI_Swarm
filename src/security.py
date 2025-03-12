from sklearn.ensemble import IsolationForest
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

class BiasDetector:
    def __init__(self):
        pass

    def detect_bias(self, data: dict):
        region_counts = {}
        for key in data.items():
            region = key.split("_")[0]  # Assume keys are like "North_sensor1"
            region_counts[region] = region_counts.get(region, 0) + 1

        total = sum(region_counts.values())
        bias_detected = any(count / total > 0.7 for count in region_counts.values())  # Arbitrary threshold
        if bias_detected:
            print(f"Bias detected in data: {region_counts}")
        else:
            print("No significant bias detected.")
        return bias_detected

class SecurityLayer:
    def __init__(self):
        self.threat_detector = IsolationForest(contamination=0.01)
        self.bias_detector = BiasDetector()
        self.bayesian_network = None

    def detect_threats(self, data: np.ndarray):
        predictions = self.threat_detector.fit_predict(data)
        threats = [i for i, pred in enumerate(predictions) if pred == -1]
        if threats:
            print(f"Security Layer detected threats at indices: {threats}")
        else:
            print("No threats detected.")
        return threats

    def enforce_access_control(self, role: str, action: str) -> bool:
        allowed_actions = {
            "admin": ["read", "write", "delete"],
            "user": ["read"],
        }
        if action in allowed_actions.get(role, []):
            print(f"Access granted for role '{role}' to perform action '{action}'.")
            return True
        print(f"Access denied for role '{role}' to perform action '{action}'.")
        return False
    
    @staticmethod
    def federated_averaging(models):
        avg_model = {}
        for key in models[0].keys():
            avg_model[key] = sum(model[key] for model in models) / len(models)
        print("Federated averaging completed successfully.")
        return avg_model
    
    def build_bayesian_network(self):
        self.bayesian_network = BayesianNetwork([('A', 'B'), ('B', 'C')])
        cpd_a = TabularCPD('A', 2, [[0.6], [0.4]])
        self.bayesian_network.add_cpds(cpd_a)
        print("Bayesian Network built successfully.")

    def probabilistic_reasoning(self, evidence: dict):
        if not self.bayesian_network:
            print("Bayesian Network not initialized.")
            return

        infer = VariableElimination(self.bayesian_network)
        result = infer.query(variables=['C'], evidence=evidence)
        print(f"Probabilistic reasoning result: {result}")

    def check_for_bias(self, data: dict):
        self.bias_detector.detect_bias(data)
