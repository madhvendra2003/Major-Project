import math
from datetime import datetime

class RSULogic:
    def __init__(self, rsu_id, decay_rate=0.1, reward_factor=0.1, penalty_factor=0.5):
        self.rsu_id = rsu_id
        self.decay_rate = decay_rate
        self.reward_factor = reward_factor
        self.penalty_factor = penalty_factor
        self.trust_db = {}
        self.last_seen = {}

    def update_trust(self, vehicle_id, behavior, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()

        if vehicle_id not in self.trust_db:
            self.trust_db[vehicle_id] = 50.0
            self.last_seen[vehicle_id] = timestamp
            print(f"[{self.rsu_id}] Initialized trust for vehicle {vehicle_id}.")
        else:
            last_time = self.last_seen[vehicle_id]
            hours_passed = (timestamp - last_time).total_seconds() / 3600.0
            decay_factor = math.exp(-self.decay_rate * hours_passed)
            self.trust_db[vehicle_id] *= decay_factor
            print(f"[{self.rsu_id}] Decayed trust for {vehicle_id} by factor {decay_factor:.4f} after {hours_passed:.2f}h.")

        self.last_seen[vehicle_id] = timestamp
        current_trust = self.trust_db[vehicle_id]

        if behavior == 1:
            increment = self.reward_factor * (100.0 - current_trust)
            current_trust += increment
            print(f"[{self.rsu_id}] Honest event: Increased trust of {vehicle_id} by {increment:.2f}.")
        else:
            current_trust *= self.penalty_factor
            print(f"[{self.rsu_id}] Malicious event: Penalized trust of {vehicle_id}.")

        current_trust = max(0.0, min(100.0, current_trust))
        self.trust_db[vehicle_id] = current_trust

        print(f"[{self.rsu_id}] Updated trust for {vehicle_id}: {round(current_trust)}")
        return int(round(current_trust))

    def get_trust_score(self, vehicle_id):
        return int(round(self.trust_db.get(vehicle_id, 50.0)))

    def get_all_trust_scores(self):
        return {vid: int(round(score)) for vid, score in self.trust_db.items()}
