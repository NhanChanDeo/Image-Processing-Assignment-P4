import torch
import torch.nn as nn

class MultiTaskLoss:
    def __init__(self, alpha=1.0, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.mae = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, age_pred, age_target, gender_logit, gender_target):
        loss_age = self.mae(age_pred, age_target)
        loss_gender = self.bce(gender_logit, gender_target)

        loss = self.alpha * loss_age + self.beta * loss_gender
        return loss, loss_age, loss_gender
