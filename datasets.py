import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ISICDataset(Dataset):
    def __init__(self, csv_file, image_dir, mode="labeled"):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mode = mode
        
        self.weak_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        self.strong_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.image_dir, img_id + ".jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.mode == "labeled" or self.mode == "val" or self.mode == "test":
            label = self.df.iloc[idx]["label"]
            if self.mode == "labeled":
                image = self.weak_transform(image)
            else:
                image = self.eval_transform(image)
            return image, label
        
        elif self.mode == "unlabeled":
            weak = self.weak_transform(image)
            strong = self.strong_transform(image)
            return weak, strong