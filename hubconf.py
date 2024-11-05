dependencies = ['torch']

def focal_loss(alpha=1.0, gamma=2.0, class_weights=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            if class_weights is not None:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            else:
                self.class_weights = None

        def forward(self, inputs, targets):
            # Move class_weights to the same device as inputs
            if self.class_weights is not None:
                self.class_weights = self.class_weights.to(inputs.device)

            # Compute log probabilities
            log_probs = F.log_softmax(inputs, dim=1)

            # Convert targets to one-hot encoding and move to the same device
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 3, 1, 2).float().to(inputs.device)

            # Compute focal loss components
            probs = torch.exp(log_probs)  # Convert log probabilities to probabilities
            focal_weight = self.alpha * (1 - probs) ** self.gamma
            focal_loss = -focal_weight * log_probs * targets_one_hot

            # Apply class weights if provided
            if self.class_weights is not None:
                focal_loss *= self.class_weights.view(1, -1, 1, 1)

            # Average over batch, channels, and spatial dimensions
            return focal_loss.mean()

    return FocalLoss(alpha=alpha, gamma=gamma, class_weights=class_weights)
