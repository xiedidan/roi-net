import numpy as np

class ScoreCounter:
    def __init__(self):
        self.scores = np.array([], dtype=np.float32)

    def add_scores(self, new_scores):
        self.scores = np.concatenate(
            [self.scores, new_scores]
        )

    def get_avg_score(self):
        avg_score = self.scores.mean()

        return avg_score
