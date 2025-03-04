from transformers import BertForSequenceClassification, Trainer, TrainingArguments

class UniversitySystem:
    def __init__(self):
        self.agents = []
        self.role_manager = RoleManager()

    def train_agent(self, agent: object, task: str):
        print(f"University System is training {agent.name} for task: {task}")
        setattr(agent, "specialization", task)

    def mentor_agent(self, mentor: object, mentee: object):
        print(f"{mentor.name} is mentoring {mentee.name}.")
        mentee.specialization = mentor.specialization

    def assign_role(self, agent: object, role: str):
        self.role_manager.assign_role(agent, role)

    def switch_role(self, agent: object, new_role: str):
        self.role_manager.switch_role(agent, new_role)

    def train_agent(self, agent: object, task: str):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        trainer = Trainer(model=model, args=TrainingArguments(output_dir="./results"))
        trainer.train()
        setattr(agent, "specialization", task)
        print(f"University System trained {agent.name} for task: {task}")

class RoleManager:
    def __init__(self):
        self.roles = {}  # Tracks current roles of agents
        self.role_hierarchy = {
            "Junior Worker": 1,
            "Senior Worker": 2,
            "Team Lead": 3,
            "Regional Manager": 4,
        }
        self.role_requirements = {
            "Junior Worker": {"training": ["Basic Task Execution"]},
            "Senior Worker": {"training": ["Advanced Task Optimization"], "experience_months": 6},
            "Team Lead": {"training": ["Leadership Training"], "experience_months": 12},
            "Regional Manager": {"training": ["Strategic Planning"], "experience_months": 24},
        }
        self.performance_metrics = {}

    def assign_role(self, agent: object, role: str):
        """
        Dynamically assigns a role to an agent if they meet the prerequisites.
        """
        if not self._validate_role_prerequisites(agent, role):
            print(f"Agent {agent.name} does not meet prerequisites for role '{role}'.")
            return

        self.roles[agent.name] = role
        print(f"Assigned role '{role}' to {agent.name}.")
        self._assign_training_if_needed(agent, role)

    def switch_role(self, agent: object, new_role: str):
        """
        Switches an agent's role dynamically after validation.
        """
        if agent.name not in self.roles:
            print(f"No role assigned to {agent.name} yet.")
            return

        if not self._validate_role_prerequisites(agent, new_role):
            print(f"Agent {agent.name} cannot switch to role '{new_role}' due to missing prerequisites.")
            return

        old_role = self.roles[agent.name]
        self.roles[agent.name] = new_role
        print(f"Switched role of {agent.name} from '{old_role}' to '{new_role}'.")
        self._assign_training_if_needed(agent, new_role)

    def _validate_role_prerequisites(self, agent: object, role: str) -> bool:
        """
        Validates if an agent meets the prerequisites for a given role.
        """
        requirements = self.role_requirements.get(role, {})
        training_completed = getattr(agent, "training_completed", [])
        experience_months = getattr(agent, "experience_months", 0)

        # Check training requirements
        if "training" in requirements:
            for course in requirements["training"]:
                if course not in training_completed:
                    print(f"Agent {agent.name} has not completed required training: {course}")
                    return False

        # Check experience requirements
        if "experience_months" in requirements:
            if experience_months < requirements["experience_months"]:
                print(f"Agent {agent.name} does not have enough experience for role '{role}'.")
                return False

        return True

    def _assign_training_if_needed(self, agent: object, role: str):
        """
        Assigns required training to an agent if they haven't completed it yet.
        """
        requirements = self.role_requirements.get(role, {})
        training_completed = getattr(agent, "training_completed", [])

        if "training" in requirements:
            for course in requirements["training"]:
                if course not in training_completed:
                    print(f"Assigning training: {course} to {agent.name}.")
                    training_completed.append(course)
            setattr(agent, "training_completed", training_completed)

    def monitor_performance(self, agent: object, task_success_rate: float):
        """
        Monitors and tracks an agent's performance in their role.
        """
        self.performance_metrics[agent.name] = task_success_rate
        print(f"Performance of {agent.name}: {task_success_rate * 100:.2f}% success rate.")

        # Provide feedback based on performance
        if task_success_rate < 0.7:
            print(f"{agent.name} needs improvement in their current role.")
        elif task_success_rate > 0.9:
            print(f"{agent.name} is excelling and may be eligible for a higher role.")

    def resolve_role_conflict(self, agents: list, role: str):
        """
        Resolves conflicts when multiple agents compete for the same role.
        """
        eligible_agents = []
        for agent in agents:
            if self._validate_role_prerequisites(agent, role):
                eligible_agents.append(agent)

        if not eligible_agents:
            print(f"No eligible agents for role '{role}'.")
            return

        # Select the most experienced agent
        selected_agent = max(eligible_agents, key=lambda agent: getattr(agent, "experience_months", 0))
        self.assign_role(selected_agent, role)
        print(f"Resolved conflict: Assigned role '{role}' to {selected_agent.name}.")

    def renew_role(self, agent: object, role: str, duration_months: int = 6):
        """
        Renews a temporary role for an agent.
        """
        if agent.name not in self.roles or self.roles[agent.name] != role:
            print(f"Agent {agent.name} is not currently assigned to role '{role}'.")
            return

        print(f"Renewed role '{role}' for {agent.name} for {duration_months} months.")
    

