import os
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from albumentations.pytorch import ToTensorV2
import wandb
import torch.nn.functional as F
import zipfile
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from typing import Optional, Callable, Tuple
from pathlib import Path
import shutil

# Configuration
config_dict = {
    # Data
    "DATASET": "ARAS400k",
    "NUM_CLASSES": 7,
    "IMAGE_EXTENSIONS": ('.png'),

    # Training
    "EPOCHS": 20,
    "BATCH_SIZE": 24,
    "PATIENCE": 5,
    "LEARNING_RATE": 1e-3,
    "GRAD_CLIP": 1.0,

    # Model
    "ARCHITECTURE": "Segformer",
    "ENCODER": "efficientnet-b7",
    "ENCODER_WEIGHTS": "imagenet",

    # DataLoader
    "NUM_WORKERS": 16,
    "PIN_MEMORY": True,
}

print(config_dict)

COLOR2LABEL = {
    (0, 100, 0): 0,  # Tree
    (255, 182, 193): 1,  # Shrub
    (154, 205, 50): 2,  # Grass
    (255, 215, 0): 3,  # Crop
    (139, 69, 19): 4,  # Built-up
    (211, 211, 211): 5,  # Barren
    (0, 0, 255): 6,  # Water
}

LABEL2COLOR = {k: list(v) for k, v in enumerate(COLOR2LABEL.keys())}

val_transforms = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
train_transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])

def mask_to_class(mask: np.ndarray) -> np.ndarray:
    """Converts an RGB mask into a single-channel class mask."""
    h, w, _ = mask.shape
    mask_class = np.zeros((h, w), dtype=np.uint8)
    for color, label in COLOR2LABEL.items():
        matches = np.all(mask == np.array(color, dtype=np.uint8), axis=-1)
        mask_class[matches] = label
    return mask_class


def class_to_mask(mask_class: np.ndarray) -> np.ndarray:
    """Converts a class index mask back into an RGB mask."""
    h, w = mask_class.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL2COLOR.items():
        rgb_mask[mask_class == label] = color
    return rgb_mask


def zip_folder(folder_path, zip_path):
    """Zip a folder."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))


class SegmentationDataset(Dataset):
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 transform: Optional[Callable] = None,
                 debug: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.debug = debug

        # Validate directories
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        # Get file lists
        self.images = self._get_sorted_files(self.image_dir)
        self.masks = self._get_sorted_files(self.mask_dir)

        # Match images and masks by filename
        self._validate_and_match_files()

        if debug:
            print(f"{image_dir}: Found {len(self.images)} images and {len(self.masks)} masks")

    def _get_sorted_files(self, directory: Path) -> list:
        """Get sorted list of image files from directory."""
        files = [f for f in directory.iterdir()
                 if f.suffix.lower() in config_dict["IMAGE_EXTENSIONS"]]
        return sorted(files, key=lambda x: x.name)

    def _validate_and_match_files(self):
        """Validate that images and masks match by filename."""
        # Create mapping of stem to full path
        image_map = {f.stem: f for f in self.images}
        mask_map = {f.stem: f for f in self.masks}

        # Find common base names
        common_names = sorted(set(image_map.keys()) & set(mask_map.keys()))

        if not common_names:
            available_images = list(image_map.keys())[:5]
            available_masks = list(mask_map.keys())[:5]
            raise ValueError(
                f"No matching image-mask pairs found!\n"
                f"Sample images: {available_images}\n"
                f"Sample masks: {available_masks}"
            )

        # Use only matching pairs
        self.images = [image_map[name] for name in common_names]
        self.masks = [mask_map[name] for name in common_names]

        if len(self.images) != len(self.masks):
            raise ValueError(f"Image count ({len(self.images)}) != Mask count ({len(self.masks)})")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(mask_path))
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask_to_class(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            # Ensure mask is long tensor after transform
            mask = mask.long()
        else:
            # Basic normalization and conversion
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()  # Explicitly make it long

        return image, mask


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        num_batches = len(loader)

        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config_dict["GRAD_CLIP"])
            self.optimizer.step()

            total_loss += loss.item()

            # Print progress every 1000of batches
            if (batch_idx + 1) % 1000 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        return total_loss / num_batches


def evaluate(model, loader, num_classes, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = DiceLoss(mode="multiclass")
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            # Calculate loss for monitoring
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())

            # Print progress every 1000 of batches
            if (batch_idx + 1) % 1000 == 0:
                print(f"  Eval Batch {batch_idx + 1}/{num_batches}")

    # Flatten arrays for sklearn metrics
    y_pred = np.concatenate([arr.ravel() for arr in all_preds])
    y_true = np.concatenate([arr.ravel() for arr in all_labels])

    metrics = {
        "accuracy": (y_pred == y_true).mean(),
        "iou": jaccard_score(y_true, y_pred, average='macro', zero_division=0),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "loss": total_loss / len(loader)
    }

    # Per-class IoU
    per_class_iou = jaccard_score(y_true, y_pred, average=None, zero_division=0)
    for i, iou in enumerate(per_class_iou):
        metrics[f"iou_class_{i}"] = iou
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, f1 in enumerate(per_class_f1):
        metrics[f"f1_class_{i}"] = f1

    return metrics

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize WandB
    wandb.init(
        project="WANDB_PROJECT",  # Replace with your WandB project name
        entity="WANDB_ENTITY",  # Replace with your WandB entity
        config=config_dict
    )

    try:
        # Create datasets
        train_ds = SegmentationDataset(
            "ARAS400k/train/images",
            "ARAS400k/train/masks",
            train_transforms,
            debug=True)
        # It is possible to concatenate multiple datasets (synth and real) if needed:
        # train_ds_2 = SegmentationDataset("ARAS400k/train/images","ARAS400k/train/masks",train_transforms,debug=True)
        # train_ds_3 = SegmentationDataset("ARAS400k/synth/images","ARAS400k/synth/masks",train_transforms,debug=True)
        # train_ds = ConcatDataset([train_ds_2, train_ds_3])
        val_ds = SegmentationDataset(
            "ARAS400k/val/images",
            "ARAS400k/val/masks",
            val_transforms,
            debug=True
        )
        test_ds = SegmentationDataset(
            "ARAS400k/test/images",
            "ARAS400k/test/masks",
            val_transforms,
            debug=True
        )

        print(f"Training samples: {len(train_ds)}")
        print(f"Validation samples: {len(val_ds)}")
        print(f"Test samples: {len(test_ds)}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        return

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=config_dict["BATCH_SIZE"], shuffle=True,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"]
    )
    val_loader = DataLoader(
        val_ds, batch_size=config_dict["BATCH_SIZE"], shuffle=False,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"]
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=config_dict["NUM_WORKERS"], pin_memory=config_dict["PIN_MEMORY"]
    )

    # Initialize model
    model = smp.Segformer(
        encoder_name=config_dict["ENCODER"],
        encoder_weights=config_dict["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config_dict["NUM_CLASSES"]
    ).to(device)

    print(f"Model: {config_dict['ARCHITECTURE']} with {config_dict['ENCODER']}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = DiceLoss(mode="multiclass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_dict["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.1
    )

    trainer = Trainer(model, criterion, optimizer, scheduler, device)

    # Training state
    best_f1 = 0.0
    counter = 0
    model_name = f"{config_dict['DATASET']}_{config_dict['ARCHITECTURE']}_{config_dict['ENCODER']}"


    print("\nStarting training...")

    # Training loop
    for epoch in range(config_dict["EPOCHS"]):
        print(f"\nEpoch {epoch + 1}/{config_dict['EPOCHS']}")

        # Train
        print("Training...")
        train_loss = trainer.train_epoch(train_loader)

        # Validate
        print("Validating...")
        val_metrics = evaluate(model, val_loader, config_dict["NUM_CLASSES"], device)
        current_lr = optimizer.param_groups[0]['lr']

        # Update scheduler
        scheduler.step(val_metrics["f1"])

        # Log metrics
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": current_lr,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        wandb.log(log_data)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}, Val IoU: {val_metrics['iou']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            counter = 0
            # Save only model state_dict for weights_only loading
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            counter += 1
            print(f"No improvement for {counter}/{config_dict['PATIENCE']} epochs")
            if counter >= config_dict["PATIENCE"]:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    # Test evaluation
    print("\n" + "=" * 50)
    print("Testing Best Model")
    print("=" * 50)

    # Fix: Load with weights_only=False since we're saving the full training state
    # or save only the model state_dict (as done above)
    model.load_state_dict(torch.load(f"best_{model_name}.pth", weights_only=True))

    print("Running test evaluation...")
    test_metrics = evaluate(model, test_loader, config_dict["NUM_CLASSES"], device)

    print("\nTest Results:")
    for k, v in test_metrics.items():
        if k.startswith('iou_class'):
            continue
        print(f"  {k}: {v:.4f}")
    for k, v in test_metrics.items():
        if k.startswith('f1_class'):
            continue
        print(f"  {k}: {v:.4f}")        

    wandb.log({f"test_{k}": v for k, v in test_metrics.items()})

    # Save predictions
    save_predictions(model, test_ds, test_loader, model_name, device)

    wandb.finish()
    print("Training completed!")


def save_predictions(model, dataset, loader, model_name, device):
    """Save prediction visualizations."""
    pred_dir = Path(model_name) / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    num_samples = len(loader)

    print("Saving predictions...")
    with torch.no_grad():
        for idx, (img, mask) in enumerate(loader):
            img = img.to(device)
            output = model(img)
            pred = torch.argmax(F.softmax(output, dim=1), dim=1)
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)

            # Convert to RGB for visualization
            rgb_pred = class_to_mask(pred)

            # Save prediction
            fname = Path(dataset.images[idx]).name
            cv2.imwrite(str(pred_dir / f"pred_{fname}"),
                        cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2BGR))

            # Print progress every 10 samples
            if (idx + 1) % 1000 == 0 or (idx + 1) == num_samples:
                print(f"  Saved {idx + 1}/{num_samples} predictions")

    # Create zip file
    zip_folder(str(pred_dir), f"{model_name}.zip")
    print(f"Predictions saved to {model_name}.zip")
    shutil.rmtree(pred_dir)
    print("Predictions folder cleaned")
    
if __name__ == "__main__":
    main()