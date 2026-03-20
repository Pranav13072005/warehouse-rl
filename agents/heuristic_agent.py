import numpy as np
from env.demand_generator import DemandGenerator

class FrequencyHeuristicAgent:
    """
    Greedy slotting heuristic: sort SKUs by demand frequency and
    assign most popular to closest slots (lowest Manhattan distance
    from depot). Re-executes one swap per step to converge toward
    this optimal static arrangement.
    """

    def __init__(self, env):
        self.env = env
        self.n_slots = env.n_slots
        self.grid_size = env.grid_size
        # Pre-compute slot distances from depot
        self.slot_distances = np.array([
            (r + c) for r in range(env.grid_size)
            for c in range(env.grid_size)
        ])
        self.sorted_slots = np.argsort(self.slot_distances)  # closest first
        # Get demand weights from generator
        self.demand_weights = env.demand_gen.get_demand_weights()
        self.sorted_skus = np.argsort(self.demand_weights)[::-1]  # most pop first

    def predict(self, obs, deterministic=True):
        """
        Find the highest-value swap: find a popular SKU in a far slot
        and swap it with a less-popular SKU in a near slot.
        """
        slot_contents = self.env.slot_contents.copy()
        best_action = 0
        best_gain = -np.inf

        # Check top candidate pairs
        for i, near_slot in enumerate(self.sorted_slots[:10]):
            for j, far_slot in enumerate(self.sorted_slots[::-1][:10]):
                if near_slot == far_slot:
                    continue
                sku_near = slot_contents[near_slot]
                sku_far = slot_contents[far_slot]
                if sku_near < 0 or sku_far < 0:
                    continue
                # Gain = putting a more popular SKU closer
                gain = (self.demand_weights[sku_far] - self.demand_weights[sku_near]) * \
                       (self.slot_distances[far_slot] - self.slot_distances[near_slot])
                if gain > best_gain:
                    best_gain = gain
                    best_action = near_slot * self.n_slots + far_slot

        return best_action, None