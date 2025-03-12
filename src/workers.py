import gymnasium as gym
import numpy as np
import torch
from torch_geometric.nn import GCNConv
from stable_baselines3 import PPO
from learn2learn.algorithms import MAML

class Worker:
    def __init__(self, name: str, region: str):
        """
        Initializes a worker agent with a unique name and region.
        """
        self.name = name
        self.region = region
        self.local_model = None
        self.meta_model = None
        self.gnn_model = self._build_gnn()
        print(f"Worker ({self.name}) initialized in region {self.region}.")

    def _build_gnn(self):
        """
        Builds a Graph Neural Network (GNN) for modeling agent interactions.
        This GNN is used for collaborative decision-making among workers.
        """
        class GNN(torch.nn.Module):
            def __init__(self, input_dim=16, hidden_dim=32, output_dim=8):
                super(GNN, self).__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        print(f"Built GNN for worker ({self.name}) collaboration.")
        return GNN()

    def collaborate_with_gnn(self, peers: list):
        """
        Collaborates with other workers using a Graph Neural Network (GNN).
        The GNN models relationships between agents to improve collective decision-making.
        """
        if not peers:
            print(f"Worker ({self.name}) has no peers to collaborate with.")
            return

        try:
            # Simulate node features for self and peers
            num_agents = len(peers) + 1  # Include self
            x = torch.randn(num_agents, 16)  # Node features (self + peers)
            edge_index = self._generate_edge_index(num_agents)

            # Pass data through the GNN
            output = self.gnn_model(x, edge_index)
            print(f"Worker ({self.name}) collaborated using GNN with output shape: {output.shape}")
        except Exception as e:
            print(f"Error during GNN collaboration: {e}")

    def _generate_edge_index(self, num_agents: int):
        """
        Generates an edge index for the GNN based on the number of agents.
        This simulates a fully connected graph where all agents are interconnected.
        """
        edges = [(i, j) for i in range(num_agents) for j in range(num_agents) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def adapt_to_task(self, task_data: list):
        """
        Adapts to new tasks using Model-Agnostic Meta-Learning (MAML).
        This allows the worker to generalize quickly to unseen tasks.
        """
        try:
            base_model = PPO("MlpPolicy", "CustomEnv", verbose=1)
            self.meta_model = MAML(base_model, lr=0.01)
            self.meta_model.adapt(task_data)
            print(f"Worker ({self.name}) adapted to new task using MAML.")
        except Exception as e:
            print(f"Error during MAML adaptation: {e}")

    def execute_task(self, task: str):
        """
        Executes a given task in the worker's region.
        """
        print(f"Worker ({self.name}) in region {self.region} is executing task: {task}")

    def optimize_locally(self, environment: str = "CustomEnv"):
        """
        Optimizes the worker's local model using reinforcement learning (RL).
        The environment can be customized for specific use cases.
        """
        try:
            env = gym.make(environment)
            self.local_model = PPO("MlpPolicy", env, verbose=1)
            self.local_model.learn(total_timesteps=10000)
            print(f"Worker ({self.name}) optimized locally using RL in environment: {environment}.")
        except Exception as e:
            print(f"Error during local optimization: {e}")

    def quantify_uncertainty(self, predictions: np.ndarray) -> float:
        """
        Quantifies uncertainty in predictions using standard deviation.
        This helps in understanding the reliability of the worker's decisions.
        """
        try:
            uncertainty = np.std(predictions)
            print(f"Worker ({self.name}) quantified uncertainty: {uncertainty:.4f}")
            return uncertainty
        except Exception as e:
            print(f"Error during uncertainty quantification: {e}")
            return 0.0
        
if __name__ == "__main__":
    worker1 = Worker(name="TrafficAgent1", region="North")
    worker2 = Worker(name="TrafficAgent2", region="South")
    worker1.collaborate_with_gnn([worker2])

    task_data = [np.random.rand(4) for _ in range(10)]  # Simulated task data
    worker1.adapt_to_task(task_data)
    worker1.optimize_locally(environment="CustomEnv")
    worker1.execute_task("Optimize traffic light timings")

    predictions = np.random.rand(10)  # Simulated predictions
    worker1.quantify_uncertainty(predictions)
