import heapq
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pika
from stable_baselines3 import PPO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s:%(funcName)s - %(message)s"
)

class EmergencyChannel:
    def __init__(self, rabbitmq_host: str = "localhost"):
        """
        Initializes the Emergency Channel system.
        """
        self.active_incidents: list = []  # Priority queue for incidents
        self.incident_details: Dict[str, Dict[str, Any]] = {}  # Detailed metadata for incidents
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="emergency_queue")

    def escalate_issue(self, issue: str, priority: int, severity: str, assigned_agent: Optional[str] = None):
        """
        Escalates critical issues to the Crisis Management Tier with a priority level and severity.
        Notifies higher-tier agents via RabbitMQ.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        incident_metadata = {
            "priority": priority,
            "severity": severity,
            "status": "Active",
            "timestamp": timestamp,
            "assigned_agent": assigned_agent,
            "resolution_time": None
        }
        self.incident_details[issue] = incident_metadata
        heapq.heappush(self.active_incidents, (priority, issue))
        self.channel.basic_publish(exchange="", routing_key="emergency_queue", body=issue)
        logging.info(f"Emergency Channel escalated issue: {issue} with priority {priority}, severity {severity} at {timestamp}")

    def resolve_incident(self):
        """
        Resolves the highest-priority incident and logs its resolution time.
        Updates the incident's status and assigns a resolution time.
        """
        if self.active_incidents:
            priority, issue = heapq.heappop(self.active_incidents)
            resolution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.incident_details[issue]["status"] = "Resolved"
            self.incident_details[issue]["resolution_time"] = resolution_time
            logging.info(f"Emergency Channel resolved issue: {issue} with priority {priority} at {resolution_time}")
        else:
            logging.warning("No active incidents to resolve.")

    def generate_report(self):
        """
        Generates a report of active incidents, sorted by priority.
        """
        if not self.active_incidents:
            logging.info("No active incidents.")
            return
        logging.info("Active Incidents Report:")
        for priority, issue in sorted(self.active_incidents):
            details = self.incident_details[issue]
            logging.info(
                f"- Issue: {issue}, Priority: {priority}, Severity: {details['severity']}, "
                f"Status: {details['status']}, Reported: {details['timestamp']}"
            )

    def assign_agent(self, issue: str, agent: str):
        """
        Assigns an agent to handle a specific incident.
        """
        if issue in self.incident_details:
            self.incident_details[issue]["assigned_agent"] = agent
            logging.info(f"Assigned agent '{agent}' to issue: {issue}")
        else:
            logging.warning(f"Issue '{issue}' not found in active incidents.")

    def automate_resolution(self, environment: str):
        """
        Automates incident resolution using reinforcement learning (RL).
        """

        model = PPO("MlpPolicy", environment, verbose=1)
        model.learn(total_timesteps=1000)  # Train model to resolve issues
        logging.info("Automated resolution mechanism trained using RL.")

    def close_connection(self):
        """
        Closes the RabbitMQ connection gracefully.
        """
        self.connection.close()
        logging.info("Closed RabbitMQ connection.")


if __name__ == "__main__":
    emergency_channel = EmergencyChannel()
    emergency_channel.escalate_issue("Fire in Building A", priority=1, severity="Critical")
    emergency_channel.escalate_issue("Power Outage", priority=2, severity="High")
    emergency_channel.assign_agent("Fire in Building A", agent="Noble_1")
    emergency_channel.assign_agent("Power Outage", agent="Advisor_2")
    emergency_channel.generate_report()
    emergency_channel.resolve_incident()
    emergency_channel.generate_report()
    emergency_channel.automate_resolution(environment="CartPole-v1")
    emergency_channel.close_connection()
