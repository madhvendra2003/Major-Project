import math
from datetime import datetime

class SourceLogic:
    def __init__(self, source_id, decay_rate=0.05, reward_factor=0.1, penalty_factor=0.2):
        self.source_id = source_id
        self.decay_rate = decay_rate
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor
        self.trust_db = {}
        self.last_seen = {}

    def update_trust(self, source_id, prediction, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(timestamp)

        if source_id not in self.trust_db:
            self.trust_db[source_id] = 50.0  # Initial neutral trust
            self.last_seen[source_id] = timestamp
            print(f"[SourceLogic] Initialized trust for source '{source_id}'.")
        else:
            last_time = self.last_seen[source_id]
            days_passed = (timestamp - last_time).total_seconds() / (3600.0 * 24.0)
            decay_factor = math.exp(-self.decay_rate * days_passed)
            self.trust_db[source_id] *= decay_factor
            print(f"[SourceLogic] Decayed trust for '{source_id}' by factor {decay_factor:.4f} after {days_passed:.2f} days.")

        self.last_seen[source_id] = timestamp
        current_trust = self.trust_db[source_id]

        # Prediction: 0 for 'real' (correct/true), 1 for 'fake' (incorrect/false)
        if prediction == 0:  # Real news - CORRECT
            # Reward for correct news - increase trust significantly
            increment = self.reward_factor * (100.0 - current_trust)
            current_trust += increment
            print(f"[SourceLogic] ✓ CORRECT: Real news from '{source_id}' - Increased trust by {increment:.2f}.")
        else:  # Fake news - INCORRECT
            # Penalize for incorrect news - decrease trust significantly
            decrement = self.penalty_factor * current_trust
            current_trust -= decrement
            print(f"[SourceLogic] ✗ INCORRECT: Fake news from '{source_id}' - Decreased trust by {decrement:.2f}.")

        current_trust = max(0.0, min(100.0, current_trust))
        self.trust_db[source_id] = current_trust

        print(f"[SourceLogic] Updated trust for '{source_id}': {int(round(current_trust))}")
        return int(round(current_trust))

    def get_trust_score(self, source_id):
        return int(round(self.trust_db.get(source_id, 50.0)))