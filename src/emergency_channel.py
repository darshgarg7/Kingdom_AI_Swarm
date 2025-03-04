class EmergencyChannel:
    def __init__(self):
        self.active_incidents = []

    def escalate_issue(self, issue: str):
        """
        Escalates critical issues to the Crisis Management Tier.
        """
        self.active_incidents.append(issue)
        print(f"Emergency Channel escalated issue: {issue}")

    def resolve_incident(self, issue: str):
        """
        Resolves an active incident and removes it from the list.
        """
        if issue in self.active_incidents:
            self.active_incidents.remove(issue)
            print(f"Emergency Channel resolved issue: {issue}")
        else:
            print(f"Issue '{issue}' not found in active incidents.")
