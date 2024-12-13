import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw


class FontDataset(Dataset):
    def __init__(self, font_paths: List[str], text_samples: List[str]):
        self.font_paths = font_paths
        self.text_samples = text_samples
        self.transforms = transforms.Compose([
            transforms.Resize((64, 512)),
            transforms.ToTensor(),
        ])
        
        # Create font_to_idx mapping
        self.font_to_idx = {path: idx for idx, path in enumerate(font_paths)}
        
    def __len__(self):
        return len(self.font_paths) * len(self.text_samples)
    
    def __getitem__(self, idx):
        font_idx = idx // len(self.text_samples)
        text_idx = idx % len(self.text_samples)
        
        font_path = self.font_paths[font_idx]
        text = self.text_samples[text_idx]
        
        # Create image with rendered text
        img = Image.new('L', (512, 64), color='white')
        try:
            font = ImageFont.truetype(font_path, 48)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, font=font, fill='black')
        except Exception as e:
            print(f"Error loading font {font_path}: {e}")
            # Return a blank image if font loading fails
            pass
        
        return self.transforms(img), font_idx