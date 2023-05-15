import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.3, scale=30):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        self.W = torch.nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_normal_(self.W)
        
    def forward(self, embeddings, labels):
        cosine = self.get_cosine(embeddings)
        onehot = self.onehot_encoding(labels)
        cosine_of_target_classes = cosine[onehot == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(cosine_of_target_classes)
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
        logits = cosine + onehot * diff
        logits = self.scale * logits
        loss = F.cross_entropy(logits, labels)
        return logits, loss
        
    def get_cosine(self, embeddings):
        normalized_embeddings = F.normalize(embeddings)
        normalized_W = F.normalize(self.W)
        cosine = F.linear(normalized_embeddings, normalized_W)
        return cosine
    
    def onehot_encoding(self, labels):
        batch_size = labels.shape[0]
        onehot = torch.zeros(batch_size, self.num_classes, device=labels.device)
        onehot.scatter_(1, labels.unsqueeze(-1), 1)
        return onehot
    
    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        eps = 1e-6
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1 + eps, 1 - eps))
        return torch.cos(angles + self.margin)
        