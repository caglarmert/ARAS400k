import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import glob
import shutil

COLOR2LABEL = {
    (0, 100, 0): 0,      # Tree
    (255, 182, 193): 1,  # Shrub
    (154, 205, 50): 2,   # Grass
    (255, 215, 0): 3,    # Crop
    (139, 69, 19): 4,    # Built-up
    (211, 211, 211): 5,  # Barren
    (0, 0, 255): 6,      # Water
}

class SemanticDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = [os.path.basename(x) for x in glob.glob(os.path.join(image_dir, "*.png"))]
        self.num_classes = len(COLOR2LABEL)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def rgb_to_label(self, mask_np):
        """Converts RGB mask to (H, W) label map using the lookup table."""
        h, w, _ = mask_np.shape
        label_map = np.zeros((h, w), dtype=np.int64)
        
        for rgb, label in COLOR2LABEL.items():
            # Create a mask for all pixels matching this RGB color
            match = np.all(mask_np == np.array(rgb), axis=-1)
            label_map[match] = label
            
        return label_map

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        
        # Load Image
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        img = self.img_transform(img)

        # Load Mask and map RGB colors to Indices
        mask = Image.open(os.path.join(self.mask_dir, name)).convert("RGB")
        mask_np = np.array(mask)
        
        label_map = self.rgb_to_label(mask_np)
        label_tensor = torch.from_numpy(label_map).long()
        
        # One-hot encode: [Classes, H, W]
        mask_onehot = F.one_hot(label_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        
        return img, mask_onehot, name

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, 128, 3, padding=1), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(128, norm_nc, 3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, 3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, label_nc):
        super().__init__()
        self.learned_shortcut = fin != fout
        self.conv_0 = nn.Conv2d(fin, fout, 3, padding=1)
        self.conv_1 = nn.Conv2d(fout, fout, 3, padding=1)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fout, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
            self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)

    def forward(self, x, seg):
        x_s = self.conv_s(self.norm_s(x, seg)) if self.learned_shortcut else x
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg), 0.2))
        return x_s + dx

class Generator(nn.Module):
    def __init__(self, label_nc):
        super().__init__()
        self.fc = nn.Conv2d(label_nc, 1024, 3, padding=1)
        self.up_0 = SPADEResnetBlock(1024, 512, label_nc)
        self.up_1 = SPADEResnetBlock(512, 256, label_nc)
        self.up_2 = SPADEResnetBlock(256, 128, label_nc)
        self.up_3 = SPADEResnetBlock(128, 64, label_nc)
        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, seg):
        x = F.interpolate(seg, size=(16, 16))
        x = self.fc(x)
        x = self.up_0(x, seg)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_1(x, seg)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_2(x, seg)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_3(x, seg)
        x = F.interpolate(x, scale_factor=2)
        return torch.tanh(self.conv_img(x))

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super().__init__()
        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(input_nc, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

def denormalize(tensor):
    """Convert normalized tensor back to [0, 1] range."""
    return (tensor + 1) / 2

def save_image_tensor(tensor, path):
    """Save a tensor as an image file."""
    img = denormalize(tensor.cpu().detach())
    img = torch.clamp(img, 0, 1)
    img = transforms.ToPILImage()(img)
    img.save(path)

# --- Configuration & Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(COLOR2LABEL)

netG = Generator(num_classes).to(device)
netD = Discriminator(num_classes + 3).to(device)
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:16].to(device).eval()

optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

output_dir = "train_generation_results"
os.makedirs(output_dir, exist_ok=True)


# --- Dataset Loading ---
train_dataset = SemanticDataset("ARAS400k/train/images", "ARAS400k/train/masks")
# Training dataloader needs batches and shuffling
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
# Inference dataloader uses batch_size=1 and no shuffling to save images properly
infer_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

print(f"Starting training on {device} with {num_classes} semantic classes.")
print(f"Training on single set: {len(train_dataset)} samples.")

# --- Training Loop ---
num_epochs = 20
for epoch in range(num_epochs):
    netG.train()
    for i, (real_imgs, masks, _) in enumerate(train_dataloader):
        real_imgs, masks = real_imgs.to(device), masks.to(device)

        # Update D
        optD.zero_grad()
        fake_imgs = netG(masks)
        pred_real = netD(torch.cat([masks, real_imgs], 1))
        pred_fake = netD(torch.cat([masks, fake_imgs.detach()], 1))
        loss_D = (F.relu(1 - pred_real).mean() + F.relu(1 + pred_fake).mean())
        loss_D.backward()
        optD.step()

        # Update G
        optG.zero_grad()
        pred_fake_G = netD(torch.cat([masks, fake_imgs], 1))
        loss_G_adv = -pred_fake_G.mean()
        loss_vgg = F.l1_loss(vgg(real_imgs), vgg(fake_imgs)) * 10.0
        
        loss_G = loss_G_adv + loss_vgg
        loss_G.backward()
        optG.step()

        if i % 100 == 0:
            print(f"Epoch {epoch} [{i}/{len(train_dataloader)}] LossD: {loss_D.item():.4f} LossG: {loss_G.item():.4f}")

# Save the trained model
model_path = "trained_generator_single_set_2.pth"
torch.save(netG.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --- Generation / Inference Loop ---
print("Generating synthetic samples from the training set...")
netG.eval()
with torch.no_grad():
    for idx, (real_imgs, masks, filenames) in enumerate(infer_dataloader):
        real_imgs, masks = real_imgs.to(device), masks.to(device)
        fake_imgs = netG(masks)
        
        # Safe to grab index 0 since infer_dataloader batch_size is 1
        filename = filenames[0]
        base_name = os.path.splitext(filename)[0]
        
        # Save generated image
        gen_path = os.path.join(output_dir, f"{base_name}.png")
        save_image_tensor(fake_imgs[0], gen_path)
        
        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1}/{len(train_dataset)} samples")

print(f"All generation results saved to {output_dir}")