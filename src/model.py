import torch
import torch.nn as nn

class SimpleDetector(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SimpleDetector, self).__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes

        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),  
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, self.num_classes)
        )

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        boxes = self.regressor(features)
        class_logits = self.classifier(features)
        return boxes, class_logits
     