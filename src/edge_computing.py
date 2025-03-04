import numpy as np

class EdgeComputing:
    def __init__(self):
        self.local_data = {}

    def process_locally(self, data: dict):
        """
        Processes data locally to reduce latency.
        """
        self.local_data.update(data)
        print(f"Edge Computing processed data locally: {data}")

    def synchronize_data(self, central_data: dict):
        """
        Synchronizes local data with central storage.
        """
        central_data.update(self.local_data)
        print(f"Edge Computing synchronized data: {central_data}")
        return central_data

    def quantify_uncertainty(self, predictions: np.ndarray) -> float:
        """
        Quantifies uncertainty in predictions using standard deviation.
        """
        uncertainty = np.std(predictions)
        print(f"Edge Computing quantified uncertainty: {uncertainty:.4f}")
        return uncertainty
    