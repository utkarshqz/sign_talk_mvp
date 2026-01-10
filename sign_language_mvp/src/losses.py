import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    y = 1 → same gesture (should be close)
    y = 0 → different gesture (should be far)
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, y):
        # Euclidean distance between embeddings
        dist = torch.norm(emb1 - emb2, dim=1)

        loss_same = y * torch.pow(dist, 2)
        loss_diff = (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        return torch.mean(loss_same + loss_diff)
