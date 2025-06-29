import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import argparse
warnings.filterwarnings('ignore')

# GPU Configuration
def setup_gpu():
    """Setup GPU configuration for CLIP model"""
    print("=== GPU Configuration for CLIP ===")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Enable mixed precision for better performance
        torch.backends.cudnn.benchmark = True
        print("✅ CUDNN benchmark enabled")

    else:
        device = torch.device('cpu')
        print("⚠️  No GPU available, using CPU")

    print(f"🔧 Device: {device}")
    print("=" * 40)
    return device

# Setup device globally
DEVICE = setup_gpu()

class EEGCLIPDataset(Dataset):
    """Dataset yang menggabungkan EEG features dengan MNIST images"""
    
    def __init__(self, eeg_features, eeg_labels, mnist_dataset, clip_preprocess):
        self.eeg_features = eeg_features
        self.eeg_labels = eeg_labels
        self.mnist_dataset = mnist_dataset
        self.clip_preprocess = clip_preprocess
        
        # Create mapping from label to MNIST indices
        self.label_to_indices = {}
        for idx, (_, label) in enumerate(mnist_dataset):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.eeg_features)
    
    def __getitem__(self, idx):
        eeg_feat = self.eeg_features[idx]
        eeg_label = self.eeg_labels[idx]
        
        # Get corresponding MNIST image
        mnist_indices = self.label_to_indices.get(eeg_label.item(), [0])
        mnist_idx = np.random.choice(mnist_indices)
        mnist_img, _ = self.mnist_dataset[mnist_idx]
        
        # Convert to PIL and apply CLIP preprocessing
        if isinstance(mnist_img, torch.Tensor):
            mnist_img = Image.fromarray((mnist_img.squeeze().numpy() * 255).astype('uint8'))
        
        mnist_img = mnist_img.convert('RGB')  # CLIP expects RGB
        clip_processed = self.clip_preprocess(mnist_img)
        
        return eeg_feat, eeg_label, clip_processed, mnist_img


class EEGCLIPModel(nn.Module):
    """Main model yang menggunakan CLIP untuk brain-to-image reconstruction"""

    def __init__(self, eeg_dim=128, clip_model_name="ViT-B/32", n_classes=10, device=None):
        super().__init__()

        # Use global device if not specified
        if device is None:
            device = DEVICE
        self.device = device

        # Load pre-trained CLIP with proper device handling
        print(f"Loading CLIP model {clip_model_name} on {device}...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_dim = self.clip_model.visual.output_dim
        print(f"✅ CLIP loaded successfully, embedding dim: {self.clip_dim}")
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # EEG feature encoder (from previous EEGNet)
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.clip_dim),
            nn.LayerNorm(self.clip_dim)
        )
        
        # Class-conditional encoding (optional)
        self.use_class_conditioning = True
        if self.use_class_conditioning:
            self.class_embedding = nn.Embedding(n_classes, self.clip_dim)
            self.class_projector = nn.Sequential(
                nn.Linear(self.clip_dim * 2, self.clip_dim),
                nn.ReLU(),
                nn.Linear(self.clip_dim, self.clip_dim),
                nn.LayerNorm(self.clip_dim)
            )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def encode_eeg(self, eeg_features, class_labels=None):
        """Encode EEG features to CLIP embedding space"""
        # Project EEG to CLIP space
        eeg_emb = self.eeg_encoder(eeg_features)
        
        # Add class conditioning
        if self.use_class_conditioning and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            combined = torch.cat([eeg_emb, class_emb], dim=-1)
            eeg_emb = self.class_projector(combined)
        
        # Normalize embeddings
        eeg_emb = F.normalize(eeg_emb, dim=-1)
        return eeg_emb
    
    def encode_images(self, images):
        """Encode images using CLIP"""
        with torch.no_grad():
            img_emb = self.clip_model.encode_image(images)
            img_emb = F.normalize(img_emb, dim=-1)
        return img_emb
    
    def forward(self, eeg_features, images, class_labels=None):
        eeg_emb = self.encode_eeg(eeg_features, class_labels)
        img_emb = self.encode_images(images)
        return eeg_emb, img_emb
    
    def contrastive_loss(self, eeg_emb, img_emb):
        """Compute contrastive loss (InfoNCE)"""
        # Compute similarity matrix
        sim_matrix = torch.matmul(eeg_emb, img_emb.T) / self.temperature
        
        # Create labels (positive pairs are on diagonal)
        batch_size = sim_matrix.size(0)
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # Compute symmetric loss
        loss_e2i = F.cross_entropy(sim_matrix, labels)
        loss_i2e = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_e2i + loss_i2e) / 2


class ImageRetrieval:
    """Retrieve images berdasarkan EEG embeddings"""
    
    def __init__(self, model, mnist_dataset, clip_preprocess, device='cuda'):
        self.model = model
        self.mnist_dataset = mnist_dataset
        self.clip_preprocess = clip_preprocess
        self.device = device
        
        # Pre-compute MNIST embeddings
        self.mnist_embeddings, self.mnist_labels, self.mnist_images = self._precompute_mnist_embeddings()
        
    def _precompute_mnist_embeddings(self, batch_size=256):
        """Pre-compute CLIP embeddings untuk semua MNIST images dengan batch processing"""
        print("Pre-computing MNIST CLIP embeddings...")

        embeddings = []
        labels = []
        images = []

        # Create dataloader for batch processing
        mnist_loader = DataLoader(self.mnist_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_imgs, batch_labels) in enumerate(mnist_loader):
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx * batch_size}/{len(self.mnist_dataset)} images")

                # Process batch
                batch_clip_inputs = []
                for img in batch_imgs:
                    # Convert to PIL and preprocess
                    if isinstance(img, torch.Tensor):
                        pil_img = Image.fromarray((img.squeeze().numpy() * 255).astype('uint8'))
                    else:
                        pil_img = img

                    pil_img = pil_img.convert('RGB')
                    clip_input = self.clip_preprocess(pil_img)
                    batch_clip_inputs.append(clip_input)

                # Stack and move to device
                batch_clip_inputs = torch.stack(batch_clip_inputs).to(self.device)

                # Get CLIP embeddings for batch
                batch_emb = self.model.encode_images(batch_clip_inputs)

                # Store results (move to CPU to save GPU memory)
                embeddings.append(batch_emb.cpu())
                labels.extend(batch_labels.tolist())
                images.extend(batch_imgs)

                # Clear GPU cache periodically
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.tensor(labels)

        print(f"✅ Pre-computed {len(embeddings)} MNIST embeddings")
        return embeddings, labels, images
    
    def retrieve_images(self, eeg_features, class_labels, top_k=5):
        """Retrieve top-k most similar images untuk given EEG features"""
        self.model.eval()
        
        with torch.no_grad():
            # Encode EEG
            eeg_emb = self.model.encode_eeg(eeg_features, class_labels)
            eeg_emb = eeg_emb.cpu()
            
            # Compute similarities
            similarities = torch.matmul(eeg_emb, self.mnist_embeddings.T)
            
            # Get top-k indices
            _, top_indices = torch.topk(similarities, top_k, dim=-1)
            
            retrieved_images = []
            retrieved_labels = []
            retrieved_similarities = []
            
            for i in range(len(eeg_features)):
                batch_images = []
                batch_labels = []
                batch_sims = []
                
                for j in range(top_k):
                    idx = top_indices[i, j].item()
                    batch_images.append(self.mnist_images[idx])
                    batch_labels.append(self.mnist_labels[idx].item())
                    batch_sims.append(similarities[i, idx].item())
                
                retrieved_images.append(batch_images)
                retrieved_labels.append(batch_labels)
                retrieved_similarities.append(batch_sims)
            
            return retrieved_images, retrieved_labels, retrieved_similarities


class CheckpointManager:
    """Manage checkpoints for EEG-CLIP model"""

    def __init__(self, checkpoint_dir='./clip_checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, val_loss, val_acc, is_best=False):
        """Save checkpoint with complete state"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'clip_model_name': getattr(model, 'clip_model_name', 'ViT-B/32'),
            'eeg_dim': getattr(model, 'eeg_dim', 128),
            'n_classes': getattr(model, 'n_classes', 10)
        }

        # Save last checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'eeg_clip_last.pth')
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'eeg_clip_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ New best model saved: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Save epoch checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f'eeg_clip_epoch_{epoch:03d}.pth')
        torch.save(checkpoint, epoch_path)

        return last_path

    def load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None):
        """Load checkpoint and restore state"""
        if checkpoint_path is None:
            # Try to load best, then last
            best_path = os.path.join(self.checkpoint_dir, 'eeg_clip_best.pth')
            last_path = os.path.join(self.checkpoint_dir, 'eeg_clip_last.pth')

            if os.path.exists(best_path):
                checkpoint_path = best_path
            elif os.path.exists(last_path):
                checkpoint_path = last_path
            else:
                print("❌ No checkpoint found")
                return None

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return None

        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=model.device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✅ Checkpoint loaded: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
        return checkpoint


def train_eeg_clip_model(model, train_loader, val_loader, epochs=100, device='cuda',
                        resume_from=None, checkpoint_dir='./clip_checkpoints'):
    """Training loop untuk EEG-CLIP model dengan checkpoint support"""

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir)

    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from:
        checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, resume_from)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"🔄 Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0
        
        for batch_idx, (eeg_feat, eeg_label, clip_img, _) in enumerate(train_loader):
            eeg_feat = eeg_feat.to(device)
            eeg_label = eeg_label.to(device)
            clip_img = clip_img.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            eeg_emb, img_emb = model(eeg_feat, clip_img, eeg_label)
            
            # Compute loss
            loss = model.contrastive_loss(eeg_emb, img_emb)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy (top-1 retrieval)
            sim_matrix = torch.matmul(eeg_emb, img_emb.T)
            pred_indices = torch.argmax(sim_matrix, dim=-1)
            correct = torch.arange(len(eeg_feat), device=device)
            acc = (pred_indices == correct).float().mean()
            train_acc += acc.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for eeg_feat, eeg_label, clip_img, _ in val_loader:
                eeg_feat = eeg_feat.to(device)
                eeg_label = eeg_label.to(device)
                clip_img = clip_img.to(device)
                
                eeg_emb, img_emb = model(eeg_feat, clip_img, eeg_label)
                loss = model.contrastive_loss(eeg_emb, img_emb)
                
                val_loss += loss.item()
                
                # Compute accuracy
                sim_matrix = torch.matmul(eeg_emb, img_emb.T)
                pred_indices = torch.argmax(sim_matrix, dim=-1)
                correct = torch.arange(len(eeg_feat), device=device)
                acc = (pred_indices == correct).float().mean()
                val_acc += acc.item()
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Temperature: {model.temperature.item():.4f}")
        print("-" * 50)

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, val_acc, is_best
        )

        # Early stopping
        if epoch > 20 and val_loss > best_val_loss * 1.1:
            print("🛑 Early stopping triggered")
            break


def evaluate_reconstruction_quality(retrieval_system, test_eeg_features, test_labels, save_path='clip_results.png'):
    """Evaluate reconstruction quality"""
    
    # Retrieve images
    retrieved_images, retrieved_labels, similarities = retrieval_system.retrieve_images(
        test_eeg_features[:20], test_labels[:20], top_k=1
    )
    
    # Calculate accuracy
    correct_retrievals = 0
    total_samples = len(test_labels[:20])
    
    for i in range(total_samples):
        true_label = test_labels[i].item()
        retrieved_label = retrieved_labels[i][0]  # Top-1 retrieval
        if true_label == retrieved_label:
            correct_retrievals += 1
    
    accuracy = correct_retrievals / total_samples
    print(f"Retrieval Accuracy: {accuracy:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    
    for i in range(min(10, total_samples)):
        # True label
        axes[0, i].text(0.5, 0.5, f'True: {test_labels[i].item()}', 
                       ha='center', va='center', fontsize=12)
        axes[0, i].set_title('True Label')
        axes[0, i].axis('off')
        
        # Retrieved image
        retrieved_img = retrieved_images[i][0]  # Top-1
        if isinstance(retrieved_img, torch.Tensor):
            retrieved_img = retrieved_img.squeeze().numpy()
        
        axes[1, i].imshow(retrieved_img, cmap='gray')
        axes[1, i].set_title(f'Retrieved: {retrieved_labels[i][0]}')
        axes[1, i].axis('off')
        
        # Similarity score
        axes[2, i].text(0.5, 0.5, f'Sim: {similarities[i][0]:.3f}', 
                       ha='center', va='center', fontsize=10)
        axes[2, i].set_title('Similarity')
        axes[2, i].axis('off')
        
        # Correctness
        correct = test_labels[i].item() == retrieved_labels[i][0]
        color = 'green' if correct else 'red'
        axes[3, i].text(0.5, 0.5, 'Correct' if correct else 'Wrong', 
                       ha='center', va='center', fontsize=10, color=color)
        axes[3, i].set_title('Result')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, similarities


def main_clip_pipeline(eeg_features_train, eeg_labels_train,
                      eeg_features_test, eeg_labels_test,
                      epochs=50, batch_size=64, resume_from=None):
    """Main pipeline untuk EEG-CLIP reconstruction dengan GPU dan checkpoint support"""

    # Use global device
    device = DEVICE
    print(f"🚀 Starting EEG-CLIP pipeline on {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    print(f"📊 MNIST dataset loaded: {len(mnist_dataset)} images")

    # Initialize model with proper device
    print("🧠 Initializing EEG-CLIP model...")
    model = EEGCLIPModel(eeg_dim=128, device=device).to(device)

    # Enable mixed precision if on GPU
    if device.type == 'cuda':
        print("⚡ Mixed precision training enabled")

    # Create datasets
    train_dataset = EEGCLIPDataset(eeg_features_train, eeg_labels_train,
                                  mnist_dataset, model.clip_preprocess)

    # Optimize batch size for GPU
    if device.type == 'cuda':
        batch_size = min(batch_size * 2, 128)
        print(f"🎯 GPU detected: Using batch size {batch_size}")

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )

    # Train model with checkpoint support
    print("🏋️ Training EEG-CLIP model...")
    train_eeg_clip_model(
        model, train_loader, val_loader,
        epochs=epochs, device=device, resume_from=resume_from
    )

    # Initialize retrieval system
    print("🔍 Initializing image retrieval system...")
    retrieval_system = ImageRetrieval(model, mnist_dataset, model.clip_preprocess, device)

    # Evaluate
    print("📊 Evaluating reconstruction quality...")
    accuracy, similarities = evaluate_reconstruction_quality(
        retrieval_system, eeg_features_test, eeg_labels_test
    )

    return model, retrieval_system, accuracy


if __name__ == "__main__":
    # Dummy data untuk testing
    # Replace dengan actual EEG features dari pipeline sebelumnya
    
    eeg_features_train = torch.randn(1000, 128)  # 1000 samples, 128 features
    eeg_labels_train = torch.randint(0, 10, (1000,))  # Random labels 0-9
    
    eeg_features_test = torch.randn(200, 128)   # 200 test samples
    eeg_labels_test = torch.randint(0, 10, (200,))
    
    # Run pipeline
    model, retrieval_system, accuracy = main_clip_pipeline(
        eeg_features_train, eeg_labels_train,
        eeg_features_test, eeg_labels_test
    )
    
    print(f"Final retrieval accuracy: {accuracy:.4f}")