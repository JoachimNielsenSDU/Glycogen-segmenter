import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt, colors


class CategoricalFocalLoss:
    '''
    Categorical Focal Loss for multi-class classification tasks.
    Takes softmax-ed predictions and integer class labels.
    '''

    def __init__(self, num_classes: int, ignore_class: int = -1, gamma: float = 2, alpha: float = 0.25):
        '''
        Initialize the Categorical Focal Loss object.

        Parameters:
        - num_classes: (int) Number of classes in the classification task.
        - gamma: (float) Focal loss gamma parameter.
        - alpha: (float) Weighting factor for the positive class.
        '''
        self.num_classes = num_classes
        self.ignore_class = ignore_class
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        '''
        Compute the Multi-class Dice Loss + Focal Loss cross-entropy loss.

        Parameters:
        - y_true: (torch.Tensor) True class labels (integer labels, must be converted to one-hot encoding).
        - y_pred: (torch.Tensor) Predicted probabilities of each class.

        Returns:
        - loss: (torch.Tensor) Categorical Focal Loss value.
        '''

        # Mask out the pixels with the ignore class
        mask = y_true != self.ignore_class

        # Use masked_select to apply the mask
        y_true = torch.masked_select(y_true, mask).float()
        y_pred = torch.masked_select(y_pred, mask)

        # Compute the focal loss on flattened predictions and labels
        epsilon = 1e-7

        pt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        pt = torch.clamp(pt, epsilon, 1 - epsilon)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)

        # Dice loss
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)

        # Combine the two losses
        return dice_loss + focal_loss.mean()
