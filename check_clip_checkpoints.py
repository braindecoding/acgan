#!/usr/bin/env python3
"""
CLIP Checkpoint Management Script
Lists available CLIP checkpoints and provides resume commands
"""

import os
import glob
import torch
from datetime import datetime

def check_clip_checkpoints(checkpoint_dir='./clip_checkpoints'):
    """Check available CLIP checkpoints"""
    print("=" * 60)
    print("🔍 CLIP CHECKPOINT STATUS")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("💡 No CLIP checkpoints available - training will start from scratch")
        return
    
    # Find all checkpoint files
    checkpoint_patterns = [
        os.path.join(checkpoint_dir, "*.pth"),
        os.path.join(checkpoint_dir, "best.pth"),
        os.path.join(checkpoint_dir, "last.pth"),
        os.path.join(checkpoint_dir, "epoch_*.pth")
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    all_checkpoints = sorted(list(set(all_checkpoints)))
    
    if not all_checkpoints:
        print(f"📁 Checkpoint directory exists: {checkpoint_dir}")
        print("❌ No checkpoint files found")
        print("💡 Training will start from scratch")
        return
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"✅ Found {len(all_checkpoints)} checkpoint(s):")
    print()
    
    # Display checkpoint details
    best_checkpoint = None
    latest_checkpoint = None
    
    for i, ckpt_path in enumerate(all_checkpoints, 1):
        filename = os.path.basename(ckpt_path)
        file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
        
        print(f"  {i}. {filename}")
        print(f"     📊 Size: {file_size:.1f} MB")
        print(f"     🕒 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to load checkpoint info
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                epoch = checkpoint.get('epoch', 'Unknown')
                val_loss = checkpoint.get('val_loss', 'Unknown')
                val_acc = checkpoint.get('val_acc', 'Unknown')
                print(f"     📈 Epoch: {epoch}")
                print(f"     📉 Val Loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"     📉 Val Loss: {val_loss}")
                print(f"     🎯 Val Acc: {val_acc:.4f}" if isinstance(val_acc, float) else f"     🎯 Val Acc: {val_acc}")
        except Exception as e:
            print(f"     ⚠️  Could not read checkpoint info: {e}")
        
        # Identify special checkpoints
        if filename == 'best.pth':
            best_checkpoint = ckpt_path
            print(f"     🏆 BEST MODEL")
        elif filename == 'last.pth':
            latest_checkpoint = ckpt_path
            print(f"     🔄 LATEST MODEL")
        
        print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if best_checkpoint:
        print(f"🏆 To resume from best checkpoint:")
        print(f"   # In your script:")
        print(f"   model, retrieval, acc = main_clip_pipeline(")
        print(f"       eeg_train, labels_train, eeg_test, labels_test,")
        print(f"       resume_from='{best_checkpoint}'")
        print(f"   )")
        print()
    
    if latest_checkpoint and latest_checkpoint != best_checkpoint:
        print(f"🔄 To resume from latest checkpoint:")
        print(f"   # In your script:")
        print(f"   model, retrieval, acc = main_clip_pipeline(")
        print(f"       eeg_train, labels_train, eeg_test, labels_test,")
        print(f"       resume_from='{latest_checkpoint}'")
        print(f"   )")
        print()
    
    if not best_checkpoint and not latest_checkpoint and all_checkpoints:
        # Use the most recent file
        most_recent = max(all_checkpoints, key=os.path.getmtime)
        print(f"📅 To resume from most recent checkpoint:")
        print(f"   model, retrieval, acc = main_clip_pipeline(")
        print(f"       eeg_train, labels_train, eeg_test, labels_test,")
        print(f"       resume_from='{most_recent}'")
        print(f"   )")
        print()
    
    print("🗑️  To start fresh (delete all checkpoints):")
    print(f"   rm -rf {checkpoint_dir}")
    print()
    
    print("=" * 60)

def clean_clip_checkpoints(checkpoint_dir='./clip_checkpoints', keep_best=True, keep_last=True):
    """Clean old CLIP checkpoints, keeping only the best and/or last"""
    print("🧹 CLEANING CLIP CHECKPOINTS...")
    
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoint directory found")
        return
    
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    
    if not all_checkpoints:
        print("❌ No checkpoints found")
        return
    
    keep_files = set()
    
    # Keep best.pth
    if keep_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        if os.path.exists(best_path):
            keep_files.add(best_path)
    
    # Keep last.pth
    if keep_last:
        last_path = os.path.join(checkpoint_dir, "last.pth")
        if os.path.exists(last_path):
            keep_files.add(last_path)
    
    # Delete other files
    deleted_count = 0
    for ckpt_path in all_checkpoints:
        if ckpt_path not in keep_files:
            try:
                os.remove(ckpt_path)
                print(f"🗑️  Deleted: {os.path.basename(ckpt_path)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {ckpt_path}: {e}")
    
    print(f"✅ Cleaned {deleted_count} checkpoint(s)")
    print(f"📁 Kept {len(keep_files)} checkpoint(s)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLIP checkpoint management")
    parser.add_argument('--checkpoint_dir', type=str, default='./clip_checkpoints', 
                       help='CLIP checkpoint directory')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean old checkpoints')
    parser.add_argument('--keep_best', action='store_true', default=True,
                       help='Keep best checkpoint when cleaning')
    parser.add_argument('--keep_last', action='store_true', default=True,
                       help='Keep last checkpoint when cleaning')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_clip_checkpoints(args.checkpoint_dir, args.keep_best, args.keep_last)
    else:
        check_clip_checkpoints(args.checkpoint_dir)
