import torch
import torch.nn as nn


class TranslatedReLU(nn.Module):
    def __init__(self, threshold=0.25, k=1):
        super(TranslatedReLU, self).__init__()
        self.threshold = threshold
        self.k = k
    
    def forward(self, predictions, labels):
        x = torch.abs(predictions - labels)
        return self.k * torch.clamp(x - self.threshold, min=0.).mean()


class SmoothK2Loss(nn.Module):
      def __init__(self, threshold=0.25, k=2):
          super(SmoothK2Loss, self).__init__()
          self.threshold = threshold
          self.k = k
      
      def forward(self, predictions, labels):
          x = torch.abs(predictions - labels)
          mask = (x >= self.threshold).type(x.dtype)
          loss = self.k * (x ** 2 - 2 * self.threshold * x + self.threshold ** 2) * mask
          return loss.mean()