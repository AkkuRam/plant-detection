import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DetectionDataset(Dataset):
    def __init__(self, data, k=9, max_samples=500):
        torch.manual_seed(42)

        self.data = data[:max_samples] if max_samples else data

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.GaussianBlur(
                kernel_size=k,
                sigma=(0.1, (k - 1) / 6)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        box = torch.tensor(item["boxes"][0], dtype=torch.float32)
        label = torch.tensor(item["class_id"], dtype=torch.long)

        return image, box, label
