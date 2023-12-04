import torch.nn as nn
import torch.nn.functional as F
import torch 

class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    Based on: https://www.baeldung.com/cs/contrastive-learning
        - L = max(0, ||x - x+||^2 - ||x - x-||^2 + m) (text)
        - L = max(0, \left \| x - x^+ \right \|^2 - \left \| x - x^- \right \|^2 + m) (LaTex)

        Args:
            - ||.|| denotes the Euclidean distance between two vectors
            - x is the anchor point input
            - x+ is the positive point input
            - x- is the negative point input
            - m is the margin (hyperparameter)

    - The goal of the loss function is to encourage the distance between the anchor point and the positive point to be smaller 
      than the distance between the anchor point and the negative point by a margin m.

    - When the distance between the anchor and positive points is not smaller than the distance between the anchor and negative points by 
      at least the margin m, the loss function is positive and encourages the model to adjust its parameters to decrease 
      the distance between the anchor and positive points and/or increase the distance between the anchor and negative points.
    """
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = F.pairwise_distance(anchor, positive, keepdim=True)
        negative_distance = F.pairwise_distance(anchor, negative, keepdim=True)

        triplet_loss = torch.mean(torch.clamp(positive_distance - negative_distance + self.margin, min=0.0))

        return triplet_loss