from stable_baselines3 import PPO
import gym
import random
from learn2learn.algorithms import MAML
import torch
from torch_geometric.nn import GCNConv

class Worker:
    def __init__(self, name: str, region: str):
        self.name = name
        self.region = region
        self.local_model = None
        self.meta_model = None
        self.gnn_model = self._build_gnn()

    def _build_gnn(self):
        """
        Builds a Graph Neural Network for modeling agent interactions.
        """
        class GNN(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(GNN, self).__init__()
                self.conv1 = GCNConv(input_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, output_dim)

            def forward(self, x, edge_index):
                x = torch.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                return x

        print("Built GNN for worker collaboration.")
        return GNN(input_dim=16, hidden_dim=32, output_dim=8)

    def collaborate_with_gnn(self, peers: list):
        """
        Collaborates with peers using a GNN to model relationships.
        """
        if not peers:
            print(f"Worker ({self.name}) has no peers to collaborate with.")
            return

        # Simulate graph data
        x = torch.randn(len(peers) + 1, 16)  # Node features (self + peers)
        edge_index = torch.tensor([[i, j] for i in range(len(peers) + 1) for j in range(len(peers) + 1)], dtype=torch.long).t()
        output = self.gnn_model(x, edge_index)
        print(f"Worker ({self.name}) collaborated using GNN with output shape: {output.shape}")

    def adapt_to_task(self, task_data: list):
        """
        Adapts to new tasks using Model-Agnostic Meta-Learning (MAML).
        """
        base_model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
        self.meta_model = MAML(base_model, lr=0.01)
        self.meta_model.adapt(task_data)
        print(f"Worker ({self.name}) adapted to new task using MAML.")

    def execute_task(self, task: str):
        print(f"Worker ({self.name}) in region {self.region} is executing task: {task}")

    def optimize_locally(self, environment: str = "CartPole-v1"):
        env = gym.make(environment)
        self.local_model = PPO("MlpPolicy", env, verbose=1)
        self.local_model.learn(total_timesteps=1000)
        print(f"Worker ({self.name}) optimized locally using RL.")

    def collaborate(self, peers: list):
        if not peers:
            print(f"Worker ({self.name}) has no peers to collaborate with.")
            return

        peer = random.choice(peers)
        print(f"Worker ({self.name}) is collaborating with {peer.name} in region {peer.region}.")
        # Share rewards to encourage cooperation
        self.reward += 10
        peer.reward += 5

    def compete(self, peers: list):
        """
        Competes with peers for resources or rewards.
        """
        if not peers:
            print(f"Worker ({self.name}) has no peers to compete with.")
            return

        peer = random.choice(peers)
        if self.reward > peer.reward:
            print(f"Worker ({self.name}) outperformed {peer.name}.")
        else:
            print(f"Worker ({self.name}) lost to {peer.name}.")

    