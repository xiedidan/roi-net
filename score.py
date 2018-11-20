import torch

class ScoreCounter:
    def __init__(self):
        self.scores = torch.tensor([], dtype=torch.float32)

    def add_scores(self, new_scores):
        self.scores = torch.cat(
            [self.scores, new_scores.cpu()]
        )

    def get_avg_score(self):
        avg_score = self.scores.mean().item()

        return avg_score
