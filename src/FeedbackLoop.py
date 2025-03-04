class FeedbackLoop:
    def __init__(self):
        self.feedback_data = []
        self.rewards = {}

    def collect_feedback(self, feedback):
        self.feedback_data.append(feedback)
        print(f"Feedback collected: {feedback}")

    def analyze_feedback(self):
        if not self.feedback_data:
            print("No feedback to analyze.")
            return
        from collections import Counter
        feedback_counts = Counter(self.feedback_data)
        print(f"Feedback analysis: {feedback_counts}")
        self.update_rewards(feedback_counts)

    def update_rewards(self, feedback_counts):
        """
        Updates rewards based on feedback performance.
        """
        for feedback, count in feedback_counts.items():
            self.rewards[feedback] = self.rewards.get(feedback, 0) + count * 10  # Reward multiplier
        print(f"Updated rewards: {self.rewards}")