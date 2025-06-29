#!/usr/bin/env python3
"""
Checkpoint Management Script for AC-GAN Project
Lists available checkpoints and provides resume commands
"""

import os
import glob
from datetime import datetime

def check_checkpoints(checkpoint_dir='./checkpoints'):
    """Check available checkpoints"""
    print("=" * 60)
    print("🔍 CHECKPOINT STATUS")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        print("💡 No checkpoints available - training will start from scratch")
        return
    
    # Find all checkpoint files
    checkpoint_patterns = [
        os.path.join(checkpoint_dir, "*.ckpt"),
        os.path.join(checkpoint_dir, "last.ckpt"),
        os.path.join(checkpoint_dir, "eeg-classifier-*.ckpt")
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
    for i, ckpt_path in enumerate(all_checkpoints, 1):
        filename = os.path.basename(ckpt_path)
        file_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(ckpt_path))
        
        print(f"  {i}. {filename}")
        print(f"     📊 Size: {file_size:.1f} MB")
        print(f"     🕒 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Extract epoch and accuracy from filename if possible
        if "epoch" in filename and "val_acc" in filename:
            try:
                parts = filename.split('-')
                epoch_part = [p for p in parts if p.startswith('epoch')][0]
                acc_part = [p for p in parts if 'val_acc' in p][0]
                epoch = epoch_part.replace('epoch', '').replace('=', '')
                acc = acc_part.replace('val_acc=', '').replace('.ckpt', '')
                print(f"     📈 Epoch: {epoch}, Val Accuracy: {acc}")
            except:
                pass
        print()
    
    # Find the best checkpoint
    best_checkpoint = None
    latest_checkpoint = None
    
    # Look for 'last.ckpt' (most recent)
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        latest_checkpoint = last_ckpt
    
    # Look for best accuracy checkpoint
    best_acc = -1
    for ckpt_path in all_checkpoints:
        if "val_acc" in ckpt_path:
            try:
                acc_str = ckpt_path.split('val_acc=')[1].split('.ckpt')[0]
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_checkpoint = ckpt_path
            except:
                continue
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    if latest_checkpoint:
        print(f"🔄 To resume from latest checkpoint:")
        print(f"   python main.py --data_path mindbigdata.txt --resume_from_checkpoint {latest_checkpoint}")
        print()
    
    if best_checkpoint and best_checkpoint != latest_checkpoint:
        print(f"🏆 To resume from best checkpoint (val_acc={best_acc:.2f}):")
        print(f"   python main.py --data_path mindbigdata.txt --resume_from_checkpoint {best_checkpoint}")
        print()
    
    if not latest_checkpoint and not best_checkpoint:
        # Use the most recent file
        most_recent = max(all_checkpoints, key=os.path.getmtime)
        print(f"📅 To resume from most recent checkpoint:")
        print(f"   python main.py --data_path mindbigdata.txt --resume_from_checkpoint {most_recent}")
        print()
    
    print("🗑️  To start fresh (delete all checkpoints):")
    print(f"   rm -rf {checkpoint_dir}")
    print()
    
    print("=" * 60)

def clean_checkpoints(checkpoint_dir='./checkpoints', keep_best=True, keep_last=True):
    """Clean old checkpoints, keeping only the best and/or last"""
    print("🧹 CLEANING CHECKPOINTS...")
    
    if not os.path.exists(checkpoint_dir):
        print("❌ No checkpoint directory found")
        return
    
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if not all_checkpoints:
        print("❌ No checkpoints found")
        return
    
    keep_files = set()
    
    # Keep last.ckpt
    if keep_last:
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            keep_files.add(last_ckpt)
    
    # Keep best checkpoint
    if keep_best:
        best_acc = -1
        best_checkpoint = None
        for ckpt_path in all_checkpoints:
            if "val_acc" in ckpt_path:
                try:
                    acc_str = ckpt_path.split('val_acc=')[1].split('.ckpt')[0]
                    acc = float(acc_str)
                    if acc > best_acc:
                        best_acc = acc
                        best_checkpoint = ckpt_path
                except:
                    continue
        if best_checkpoint:
            keep_files.add(best_checkpoint)
    
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
    
    parser = argparse.ArgumentParser(description="Checkpoint management for AC-GAN")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Checkpoint directory')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean old checkpoints')
    parser.add_argument('--keep_best', action='store_true', default=True,
                       help='Keep best checkpoint when cleaning')
    parser.add_argument('--keep_last', action='store_true', default=True,
                       help='Keep last checkpoint when cleaning')
    
    args = parser.parse_args()
    
    if args.clean:
        clean_checkpoints(args.checkpoint_dir, args.keep_best, args.keep_last)
    else:
        check_checkpoints(args.checkpoint_dir)
