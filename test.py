import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Reshape to (batch_size, k, k)
        x = x.view(-1, self.k, self.k)
        
        # Initialize as identity matrix + learned features
        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        x = x + identity
        return x

class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.input_transform = TNet(k=3)
        self.feature_transform = TNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # Input Transform
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)

        # MLP (64) -> Feature Transform
        x = F.relu(self.bn1(self.conv1(x)))

        # Feature Transform
        feature_transform = self.feature_transform(x)
        x = torch.bmm(feature_transform, x)
        local_features = x

        # MLP (128, 1024)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global features
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x, feature_transform

    def get_loss(self, pred, target, trans_feat, alpha=0.001):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, target)
        # Regularization term for the transformation matrix
        batch_size = trans_feat.size(0)
        k = trans_feat.size(1)
        identity = torch.eye(k, requires_grad=True).repeat(batch_size, 1, 1)
        if trans_feat.is_cuda:
            identity = identity.cuda()
        loss += alpha * torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - identity, dim=(1, 2)))
        return loss
