from stable_baselines3 import DQN
import hashlib
from collections import defaultdict
import numpy as np
from typing import List

class Sharding:
    def __init__(self, num_shards: int, replication_factor: int = 2):
        """
        Initializes the sharding mechanism with support for replication.
        """
        self.num_shards = num_shards
        self.replication_factor = replication_factor
        self.shards = defaultdict(list)
        self.replicas = defaultdict(list)

    def _compute_shard_id(self, data: str) -> int:
        """
        Computes the shard ID using a consistent hashing mechanism.
        """
        hash_value = int(hashlib.sha256(data.encode()).hexdigest(), 16)
        return hash_value % self.num_shards

    def assign_to_shard(self, data: str):
        """
        Assigns data to a shard and replicates it across multiple shards for fault tolerance.
        """
        shard_id = self._compute_shard_id(data)
        self.shards[shard_id].append(data)

        # Replicate data to additional shards
        replica_shards = [(shard_id + i) % self.num_shards for i in range(1, self.replication_factor + 1)]
        for replica_id in replica_shards:
            self.replicas[replica_id].append(data)

        print(f"Data '{data}' assigned to shard {shard_id} with replicas in shards {replica_shards}.")

    def get_data_from_shard(self, shard_id: int) -> list:
        """
        Retrieves data from a specific shard, including its replicas.
        """
        primary_data = self.shards.get(shard_id, [])
        replica_data = self.replicas.get(shard_id, [])
        print(f"Retrieved data from shard {shard_id}: {primary_data + replica_data}")
        return primary_data + replica_data

    def rebalance_shards(self):
        """
        Rebalances shards dynamically based on load or size.
        """
        all_data = [item for shard_data in self.shards.values() for item in shard_data]
        self.shards.clear()
        self.replicas.clear()

        for data in all_data:
            self.assign_to_shard(data)

        print("Shards rebalanced successfully.")

class LoadBalancer:
    def __init__(self, resources: List[str]):
        """
        Initializes the load balancer with RL-based optimization.
        """
        self.resources = resources
        self.model = DQN("MlpPolicy", "CartPole-v1", verbose=1)
        self.resource_usage = {resource: 0 for resource in resources}
        self.failover_resources = []

    def _get_state(self) -> np.ndarray:
        """
        Generates the current state representation for RL.
        """
        return np.array([self.resource_usage[res] for res in self.resources])

    def balance_load(self, tasks: List[str]):
        """
        Balances tasks across resources using RL.
        """
        self.model.learn(total_timesteps=10000)

        for task in tasks:
            state = self._get_state()
            action, _ = self.model.predict(state)
            resource = self.resources[action % len(self.resources)]

            # Simulate task execution and update resource usage
            self.resource_usage[resource] += 1
            print(f"Task '{task}' assigned to resource {resource} (current usage: {self.resource_usage[resource]}).")

            # Check for overloading and trigger failover if necessary
            if self.resource_usage[resource] > 80:  # Arbitrary threshold
                self.trigger_failover(resource)

    def trigger_failover(self, failed_resource: str):
        """
        Triggers failover by redistributing tasks to backup resources.
        """
        if not self.failover_resources:
            print(f"No failover resources available for {failed_resource}.")
            return

        backup_resource = self.failover_resources.pop(0)
        print(f"Failover triggered: Redirecting tasks from {failed_resource} to {backup_resource}.")
        self.resource_usage[failed_resource] = 0  # Reset usage for failed resource
        self.resource_usage[backup_resource] += self.resource_usage[failed_resource]

    def add_failover_resource(self, resource: str):
        """
        Adds a new resource to the failover pool.
        """
        self.failover_resources.append(resource)
        print(f"Added failover resource: {resource}.")

    def monitor_resource_usage(self):
        """
        Monitors and optimizes resource usage periodically.
        """
        print("Monitoring resource usage...")
        for resource, usage in self.resource_usage.items():
            if usage > 70:  # Arbitrary threshold
                print(f"Resource {resource} is overloaded (usage: {usage}). Triggering optimization.")
                self.balance_load(["dummy_task"])  # Simulate rebalancing

class CrisisManager:
    def __init__(self):
        self.backup_nodes = []
        self.failover_mechanisms = []

    def add_backup_node(self, node: str):
        self.backup_nodes.append(node)
        print(f"Backup node added: {node}")

    def activate_failover(self, failed_node: str):
        if self.backup_nodes:
            backup_node = self.backup_nodes.pop(0)
            print(f"Failover activated: Replacing {failed_node} with {backup_node}.")
        else:
            print("No backup nodes available.")

class EnergyManager:
    def __init__(self):
        self.energy_usage = {}

    def monitor_energy(self, component: str, usage: float):
        """
        Monitors energy usage of system components.
        """
        self.energy_usage[component] = usage
        print(f"Energy usage for {component}: {usage} kWh")

    def optimize_energy(self):
        """
        Optimizes energy usage by reducing unnecessary tasks.
        """
        print("Optimizing energy usage...")
        # Example: Reduce energy-intensive tasks during off-peak hours

class ScalabilityManager:
    def __init__(self, num_shards: int, resources: List[str]):
        """
        Manages both sharding and load balancing for scalability.
        """
        self.sharding = Sharding(num_shards=num_shards)
        self.load_balancer = LoadBalancer(resources=resources)

    def process_data(self, data: str, tasks: List[str]):
        """
        Processes data by assigning it to shards and balancing tasks across resources.
        """
        self.sharding.assign_to_shard(data)
        self.load_balancer.balance_load(tasks)

    def handle_failover(self, failed_resource: str):
        """
        Handles failover scenarios by redistributing tasks.
        """
        self.load_balancer.trigger_failover(failed_resource)

    def rebalance_system(self):
        """
        Rebalances both shards and resources dynamically.
        """
        self.sharding.rebalance_shards()
        self.load_balancer.monitor_resource_usage()
