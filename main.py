import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping as PLEarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal
from scipy.signal import butter, filtfilt
import mne
from collections import defaultdict
import warnings
import os
from typing import Tuple, List, Dict, Optional
import argparse
from dataclasses import dataclass

warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Configuration class for hyperparameters"""
    # Data parameters
    target_channels: List[str] = None
    sampling_rate: int = 128
    window_size: int = 32
    overlap: int = 4
    
    # Model parameters
    n_classes: int = 10
    latent_dim: int = 128
    img_size: int = 28
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    max_epochs: int = 150
    gan_epochs: int = 2000
    
    # Preprocessing parameters
    notch_freq: float = 50.0
    bandpass_low: float = 0.4
    bandpass_high: float = 60.0
    artifact_threshold: float = 100.0
    
    def __post_init__(self):
        if self.target_channels is None:
            self.target_channels = ['T7', 'P7', 'T8', 'P8']

class MindBigDataProcessor:
    """Enhanced processor for MindBigData with PyTorch integration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device_channels = {
            'MW': ['FP1'],
            'EP': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            'MU': ['TP9', 'FP1', 'FP2', 'TP10'],
            'IN': ['AF3', 'AF4', 'T7', 'T8', 'PZ']
        }
        self.sampling_rates = {'MW': 512, 'EP': 128, 'MU': 220, 'IN': 128}
        self.scaler = StandardScaler()
        
    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse single line from MindBigData format"""
        try:
            parts = line.strip().split('\t')
            if len(parts) != 7:
                return None
                
            id_val = int(parts[0])
            event = int(parts[1])
            device = parts[2]
            channel = parts[3]
            code = int(parts[4])
            size = int(parts[5])
            
            # Parse data values
            data_str = parts[6]
            if device in ['MW', 'MU']:
                data = [int(x) for x in data_str.split(',')]
            else:  # EP, IN
                data = [float(x) for x in data_str.split(',')]
                
            return {
                'id': id_val,
                'event': event,
                'device': device,
                'channel': channel,
                'code': code,
                'size': size,
                'data': np.array(data)
            }
        except Exception as e:
            return None
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from MindBigData file"""
        print(f"Loading data from {file_path}...")
        
        data_list = []
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")
                    
                parsed = self.parse_line(line)
                if parsed and parsed['code'] >= 0:  # Skip random signals (-1)
                    data_list.append(parsed)
        
        print(f"Loaded {len(data_list)} valid signals")
        return data_list
    
    def preprocess_signal(self, signal_data: np.ndarray, fs: int = 128) -> Optional[np.ndarray]:
        """Preprocess EEG signal with filtering and artifact removal"""
        
        try:
            # Notch filter for line noise
            f_notch = self.config.notch_freq
            Q = 30
            b_notch, a_notch = signal.iirnotch(f_notch, Q, fs)
            filtered_signal = signal.filtfilt(b_notch, a_notch, signal_data)
            
            # Bandpass filter
            low_freq = self.config.bandpass_low
            high_freq = self.config.bandpass_high
            nyquist = fs / 2
            low = low_freq / nyquist
            high = min(high_freq / nyquist, 0.99)
                
            b, a = butter(5, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, filtered_signal)
            
            # Artifact removal
            if np.max(np.abs(filtered_signal)) > self.config.artifact_threshold:
                return None
                
            return filtered_signal
            
        except Exception as e:
            return None
    
    def apply_car(self, signals_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply Common Average Reference"""
        channels = list(signals_dict.keys())
        signals_array = np.array([signals_dict[ch] for ch in channels])
        
        # Calculate common average
        car_signal = np.mean(signals_array, axis=0)
        
        # Subtract from each channel
        car_applied = {}
        for i, ch in enumerate(channels):
            car_applied[ch] = signals_array[i] - car_signal
            
        return car_applied
    
    def segment_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Segment signal using sliding window"""
        window_size = self.config.window_size
        overlap = self.config.overlap
        step = window_size - overlap
        
        segments = []
        for i in range(0, len(signal_data) - window_size + 1, step):
            segment = signal_data[i:i + window_size]
            segments.append(segment)
            
        return np.array(segments)
    
    def prepare_dataset(self, data_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for training"""
        print("Preparing dataset...")
        
        # Group data by event
        events = defaultdict(list)
        for item in data_list:
            if item['channel'] in self.config.target_channels:
                events[item['event']].append(item)
        
        X_data = []
        y_data = []
        
        for event_id, event_data in events.items():
            if len(event_data) < len(self.config.target_channels):
                continue
                
            # Process each channel
            channels_data = {}
            code = None
            
            for item in event_data:
                if item['channel'] in self.config.target_channels:
                    # Resample if needed
                    original_fs = self.sampling_rates.get(item['device'], 128)
                    if original_fs != self.config.sampling_rate:
                        num_samples = int(len(item['data']) * self.config.sampling_rate / original_fs)
                        resampled = signal.resample(item['data'], num_samples)
                    else:
                        resampled = item['data']
                    
                    # Preprocess
                    processed = self.preprocess_signal(resampled, self.config.sampling_rate)
                    if processed is not None:
                        channels_data[item['channel']] = processed
                        code = item['code']
            
            # Check if we have all target channels
            if len(channels_data) == len(self.config.target_channels) and code is not None:
                # Apply CAR
                car_applied = self.apply_car(channels_data)
                
                # Create multi-channel signal matrix
                signal_matrix = []
                for ch in self.config.target_channels:
                    if ch in car_applied:
                        signal_matrix.append(car_applied[ch])
                
                if len(signal_matrix) == len(self.config.target_channels):
                    signal_matrix = np.array(signal_matrix)
                    
                    # Segment signals
                    segments_per_channel = []
                    min_segments = float('inf')
                    
                    for i in range(len(self.config.target_channels)):
                        segments = self.segment_signal(signal_matrix[i])
                        segments_per_channel.append(segments)
                        min_segments = min(min_segments, len(segments))
                    
                    # Take same number of segments from each channel
                    for seg_idx in range(min_segments):
                        multi_channel_segment = []
                        for ch_idx in range(len(self.config.target_channels)):
                            multi_channel_segment.append(segments_per_channel[ch_idx][seg_idx])
                        
                        X_data.append(np.array(multi_channel_segment))
                        y_data.append(code)
        
        X_data = np.array(X_data)
        y_data = np.array(y_data)
        
        print(f"Dataset prepared: {X_data.shape}, Labels: {len(np.unique(y_data))}")
        return X_data, y_data


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label


class EEGNet(nn.Module):
    """
    EEGNet implementation in PyTorch
    Based on Lawhern et al. 2018: EEGNet: A Compact Convolutional Network for EEG-based BCIs
    """
    
    def __init__(self, n_channels: int = 4, n_classes: int = 10, 
                 samples: int = 32, dropout_rate: float = 0.25,
                 kernel_length: int = 8, F1: int = 8, D: int = 2, F2: int = 16):
        super(EEGNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.samples = samples
        
        # Block 1: Temporal convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate the size after convolutions
        self.feature_size = self._get_conv_output_size()
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            nn.Linear(128, n_classes)
        )
        
        # Feature extractor (for GAN)
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
    def _get_conv_output_size(self):
        """Calculate the output size after convolutions"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.n_channels, self.samples)
            x = self.block1(dummy_input)
            x = self.block2(x)
            return x.numel()
    
    def forward(self, x, return_features=False):
        # Add channel dimension if not present
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
            
        x = self.block1(x)
        x = self.block2(x)
        
        if return_features:
            features = self.feature_extractor(x)
            return features
        else:
            logits = self.classifier(x)
            return logits


class EEGClassifierPL(pl.LightningModule):
    """PyTorch Lightning wrapper for EEG classifier"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = EEGNet(
            n_channels=len(config.target_channels),
            n_classes=config.n_classes,
            samples=config.window_size
        )
        self.save_hyperparameters()
        
    def forward(self, x, return_features=False):
        return self.model(x, return_features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'test_loss': loss, 'test_acc': acc, 'preds': logits.argmax(dim=1), 'targets': y}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class Generator(nn.Module):
    """AC-GAN Generator"""
    
    def __init__(self, latent_dim: int = 128, n_classes: int = 10, img_size: int = 28):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = img_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(n_classes, latent_dim)
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 7 * 7 * 256)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, eeg_features, labels):
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Combine EEG features with label embeddings
        gen_input = eeg_features * label_emb
        
        # Generate image
        out = self.fc(gen_input)
        out = out.view(out.shape[0], 256, 7, 7)
        img = self.conv_blocks(out)
        
        return img


class Discriminator(nn.Module):
    """AC-GAN Discriminator"""
    
    def __init__(self, n_classes: int = 10, img_size: int = 28):
        super(Discriminator, self).__init__()
        
        self.n_classes = n_classes
        self.img_size = img_size
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25)
        )
        
        # Calculate feature size
        self.feature_size = self._get_conv_output_size()
        
        # Outputs
        self.validity_head = nn.Linear(self.feature_size, 1)
        self.class_head = nn.Linear(self.feature_size, n_classes)
        
    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.img_size, self.img_size)
            x = self.conv_blocks(dummy_input)
            return x.view(x.size(0), -1).size(1)
    
    def forward(self, img):
        features = self.conv_blocks(img)
        features = features.view(features.size(0), -1)
        
        validity = self.validity_head(features)
        label = self.class_head(features)
        
        return validity, label


class ACGAN:
    """AC-GAN implementation for EEG-to-Image reconstruction"""
    
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        self.generator = Generator(
            latent_dim=config.latent_dim,
            n_classes=config.n_classes,
            img_size=config.img_size
        ).to(device)
        
        self.discriminator = Discriminator(
            n_classes=config.n_classes,
            img_size=config.img_size
        ).to(device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
    def train_step(self, eeg_features, labels, real_images):
        batch_size = eeg_features.size(0)
        
        # Labels for adversarial loss
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.d_optimizer.zero_grad()
        
        # Real images
        real_validity, real_class = self.discriminator(real_images)
        d_real_loss = (self.adversarial_loss(real_validity, valid) + 
                      self.classification_loss(real_class, labels))
        
        # Fake images
        fake_images = self.generator(eeg_features, labels)
        fake_validity, fake_class = self.discriminator(fake_images.detach())
        d_fake_loss = (self.adversarial_loss(fake_validity, fake) + 
                      self.classification_loss(fake_class, labels))
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        self.g_optimizer.zero_grad()
        
        fake_validity, fake_class = self.discriminator(fake_images)
        g_loss = (self.adversarial_loss(fake_validity, valid) + 
                 self.classification_loss(fake_class, labels))
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'fake_images': fake_images.detach()
        }


def calculate_metrics(generated_imgs: torch.Tensor, target_imgs: torch.Tensor):
    """Calculate Dice coefficient and SSIM"""
    from skimage.metrics import structural_similarity as ssim
    
    generated_imgs = generated_imgs.cpu().numpy()
    target_imgs = target_imgs.cpu().numpy()
    
    dice_scores = []
    ssim_scores = []
    
    for i in range(len(generated_imgs)):
        gen_img = generated_imgs[i, 0]  # Remove channel dimension
        target_img = target_imgs[i, 0]
        
        # Normalize to [0, 1]
        gen_img = (gen_img + 1) / 2
        target_img = (target_img + 1) / 2
        
        # Binarize for Dice coefficient
        gen_binary = (gen_img > 0.5).astype(int)
        target_binary = (target_img > 0.5).astype(int)
        
        # Dice coefficient
        intersection = np.sum(gen_binary * target_binary)
        dice = (2.0 * intersection) / (np.sum(gen_binary) + np.sum(target_binary) + 1e-8)
        dice_scores.append(dice)
        
        # SSIM
        ssim_score = ssim(gen_img, target_img, data_range=1.0)
        ssim_scores.append(ssim_score)
    
    return np.array(dice_scores), np.array(ssim_scores)


def visualize_results(original_imgs, generated_imgs, dice_scores, ssim_scores, save_path='results.png'):
    """Visualize reconstruction results"""
    n_samples = min(10, len(original_imgs))
    
    fig, axes = plt.subplots(3, n_samples, figsize=(20, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original_imgs[i, 0], cmap='gray')
        axes[0, i].set_title(f'Original')
        axes[0, i].axis('off')
        
        # Generated
        axes[1, i].imshow(generated_imgs[i, 0], cmap='gray')
        axes[1, i].set_title(f'Generated')
        axes[1, i].axis('off')
        
        # Difference
        diff = np.abs(original_imgs[i, 0] - generated_imgs[i, 0])
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'SSIM: {ssim_scores[i]:.3f}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to MindBigData file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for classifier')
    parser.add_argument('--gan_epochs', type=int, default=2000, help='Number of epochs for GAN')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Configuration
    config = Config(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        gan_epochs=args.gan_epochs
    )
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = MindBigDataProcessor(config)
    
    # Load and prepare data
    print("Loading MindBigData...")
    data_list = processor.load_data(args.data_path)
    X, y = processor.prepare_dataset(data_list)
    
    if len(X) == 0:
        print("No valid data found!")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize data
    X_train_norm = processor.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_norm = processor.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_norm = processor.scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = EEGDataset(X_train_norm, y_train)
    val_dataset = EEGDataset(X_val_norm, y_val)
    test_dataset = EEGDataset(X_test_norm, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = EEGClassifierPL(config)
    
    # Setup trainer
    callbacks = [
        PLEarlyStopping(monitor='val_loss', patience=15),
        ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
    ]
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        callbacks=callbacks,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision=16,
        log_every_n_steps=10
    )
    
    # Train classifier
    print("Training EEG classifier...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test classifier
    test_results = trainer.test(model, test_loader)
    print(f"Test accuracy: {test_results[0]['test_acc']:.4f}")
    
    # Extract features for GAN
    model.eval()
    model = model.to(device)  # Ensure model is on GPU
    with torch.no_grad():
        train_features = []
        train_labels = []
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            features = model(x, return_features=True)
            train_features.append(features.cpu())
            train_labels.append(y)
        
        test_features = []
        test_labels = []
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            features = model(x, return_features=True)
            test_features.append(features.cpu())
            test_labels.append(y)
    
    train_features = torch.cat(train_features, dim=0).to(device)
    train_labels = torch.cat(train_labels, dim=0).to(device)
    test_features = torch.cat(test_features, dim=0).to(device)
    test_labels = torch.cat(test_labels, dim=0).to(device)
    
    print(f"Extracted features: Train {train_features.shape}, Test {test_features.shape}")
    
    # Load MNIST for real images (target domain)
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_loader = DataLoader(mnist_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Initialize AC-GAN
    print("Initializing AC-GAN...")
    acgan = ACGAN(config, device)
    
    # Training AC-GAN
    print("Training AC-GAN...")
    
    # Create combined dataset for GAN training
    class GANDataset(Dataset):
        def __init__(self, eeg_features, eeg_labels, mnist_dataset):
            self.eeg_features = eeg_features
            self.eeg_labels = eeg_labels
            self.mnist_dataset = mnist_dataset
            
        def __len__(self):
            return len(self.eeg_features)
        
        def __getitem__(self, idx):
            eeg_feat = self.eeg_features[idx]
            eeg_label = self.eeg_labels[idx]
            
            # Get corresponding MNIST image
            mnist_indices = [i for i, (_, label) in enumerate(self.mnist_dataset) if label == eeg_label.item()]
            if mnist_indices:
                mnist_idx = np.random.choice(mnist_indices)
                mnist_img, _ = self.mnist_dataset[mnist_idx]
            else:
                # Fallback to random image with same label
                mnist_img = torch.randn(1, 28, 28) * 0.5
            
            return eeg_feat, eeg_label, mnist_img
    
    gan_dataset = GANDataset(train_features.cpu(), train_labels.cpu(), mnist_dataset)
    gan_loader = DataLoader(gan_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training loop for AC-GAN
    acgan.generator.train()
    acgan.discriminator.train()
    
    for epoch in range(config.gan_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for batch_idx, (eeg_feat, eeg_label, real_img) in enumerate(gan_loader):
            eeg_feat = eeg_feat.to(device)
            eeg_label = eeg_label.to(device)
            real_img = real_img.to(device)
            
            # Train AC-GAN
            losses = acgan.train_step(eeg_feat, eeg_label, real_img)
            epoch_d_loss += losses['d_loss']
            epoch_g_loss += losses['g_loss']
        
        # Print progress
        if epoch % 100 == 0:
            avg_d_loss = epoch_d_loss / len(gan_loader)
            avg_g_loss = epoch_g_loss / len(gan_loader)
            print(f"Epoch [{epoch}/{config.gan_epochs}] D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")
            
            # Generate sample images
            with torch.no_grad():
                sample_features = test_features[:8]
                sample_labels = test_labels[:8]
                generated_samples = acgan.generator(sample_features, sample_labels)
                
                # Save sample images
                save_sample_images(generated_samples, sample_labels, epoch)
    
    # Final evaluation
    print("Evaluating reconstruction quality...")
    acgan.generator.eval()
    
    with torch.no_grad():
        # Generate images for test set
        generated_images = acgan.generator(test_features, test_labels)
        
        # Get corresponding MNIST images for comparison
        test_mnist_images = []
        for label in test_labels:
            # Find MNIST images with the same label
            mnist_indices = [i for i, (_, mnist_label) in enumerate(mnist_dataset) if mnist_label == label.item()]
            if mnist_indices:
                idx = np.random.choice(mnist_indices)
                img, _ = mnist_dataset[idx]
                test_mnist_images.append(img)
            else:
                # Create placeholder
                test_mnist_images.append(torch.zeros(1, 28, 28))
        
        test_mnist_images = torch.stack(test_mnist_images).to(device)
        
        # Calculate metrics
        dice_scores, ssim_scores = calculate_metrics(generated_images, test_mnist_images)
        
        print(f"\nReconstruction Results:")
        print(f"Average Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
        print(f"Average SSIM Score: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
        
        # Per-class results
        print(f"\nPer-class results:")
        for class_idx in range(config.n_classes):
            class_mask = (test_labels.cpu() == class_idx).numpy()
            if np.any(class_mask):
                class_dice = np.mean(dice_scores[class_mask])
                class_ssim = np.mean(ssim_scores[class_mask])
                print(f"Class {class_idx}: Dice={class_dice:.4f}, SSIM={class_ssim:.4f}")
        
        # Visualize results
        n_viz = min(10, len(generated_images))
        
        # Convert to numpy for visualization
        orig_np = test_mnist_images[:n_viz].cpu().numpy()
        gen_np = generated_images[:n_viz].cpu().numpy()
        
        # Denormalize images (from [-1,1] to [0,1])
        orig_np = (orig_np + 1) / 2
        gen_np = (gen_np + 1) / 2
        
        visualize_results(orig_np, gen_np, dice_scores[:n_viz], ssim_scores[:n_viz], 
                         save_path='reconstruction_results.png')
        
        # Save detailed results
        results_df = pd.DataFrame({
            'true_label': test_labels.cpu().numpy(),
            'dice_score': dice_scores,
            'ssim_score': ssim_scores
        })
        results_df.to_csv('detailed_results.csv', index=False)
        print("Detailed results saved to detailed_results.csv")
        
        # Generate classification report for the classifier
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        print("\nEEG Classification Report:")
        print(classification_report(all_labels, all_preds))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(config.n_classes), 
                   yticklabels=range(config.n_classes))
        plt.title('EEG Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_sample_images(generated_imgs, labels, epoch, save_dir='./samples'):
    """Save sample generated images during training"""
    import os
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Denormalize images
    generated_imgs = (generated_imgs + 1) / 2
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(8, len(generated_imgs))):
        axes[i].imshow(generated_imgs[i, 0].cpu().numpy(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/generated_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    """Create PyTorch data loaders"""
    
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main pipeline
    main()