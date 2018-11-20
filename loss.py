import torch
import torch.nn.functional as F

def compute_losses(predictions, targets):
    class_preds, score_preds = predictions
    class_tgts, score_tgts = targets

    class_loss = F.cross_entropy(class_preds, class_tgts)
    score_loss = F.l1_loss(score_preds, score_tgts)

    return (class_loss, score_loss)
