import numpy as np

class ExpertScheduler:
    def __init__(self, n_experts, expected_freq=3, window_size=3,warmstart=10):
        self.n_experts = n_experts
        self.expected_freq = expected_freq
        self.window_size = window_size
        self.total_len = np.zeros((self.n_experts, self.window_size), dtype=np.float32)
        self.cnts = np.zeros(self.n_experts)
        self.cur = 0.0 # [0,1]
        self.warmstart = warmstart

    def record_eps_len(self,length, expert_id):
        # Sliding window always overriding the earliest one
        self.total_len[expert_id, int(self.cnts[expert_id])] = length
        self.cnts[expert_id] += 1
        if self.cnts[expert_id] == self.window_size:
            self.cnts[expert_id] = 0

    def update(self, expert_id, iter):
        if iter <= self.warmstart:
            if iter % self.expected_freq == 0:
                return True
            return False
        else:
            cum_diff = 210 - self.total_len.mean(axis=1)
            cum_diff = (cum_diff/cum_diff.sum()).cumsum()
            self.cur += 1 / (self.n_experts * self.expected_freq)
            print("Schedule:",cum_diff, self.cur)
            if self.cur >= cum_diff[expert_id]:
                if self.cur >= 1:
                    self.cur = 0.0
                return True
            return False
        