import torch
import os
import sys


def get_size_mb(obj):
    """Calculate size of a PyTorch object in MB"""
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement() / (1024 ** 2)
    elif isinstance(obj, dict):
        return sum(get_size_mb(v) for v in obj.values())
    else:
        # For non-tensor objects, try to estimate
        try:
            import pickle
            return len(pickle.dumps(obj)) / (1024 ** 2)
        except:
            return 0


def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint file and print detailed information"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: File not found: {checkpoint_path}")
        return

    # Get file size
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    print(f"\n{'='*80}")
    print(f"Checkpoint Analysis: {os.path.basename(checkpoint_path)}")
    print(f"{'='*80}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Full path: {checkpoint_path}")
    print(f"\n{'='*80}")
    print("Contents:")
    print(f"{'='*80}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Analyze each key
    total_analyzed = 0
    for key in sorted(checkpoint.keys()):
        value = checkpoint[key]

        if isinstance(value, dict):
            # For nested dicts (like model, optimizer, scheduler)
            size_mb = get_size_mb(value)
            num_items = len(value)
            print(f"\n[{key}]")
            print(f"  Type: dict with {num_items} items")
            print(f"  Size: {size_mb:.2f} MB ({size_mb/file_size_mb*100:.1f}%)")

            # Show a few sample keys
            sample_keys = list(value.keys())[:5]
            if len(sample_keys) > 0:
                print(f"  Sample keys: {', '.join(str(k) for k in sample_keys)}")
                if len(value) > 5:
                    print(f"               ... and {len(value) - 5} more")

            total_analyzed += size_mb

        elif isinstance(value, torch.Tensor):
            size_mb = get_size_mb(value)
            print(f"\n[{key}]")
            print(f"  Type: Tensor")
            print(f"  Shape: {value.shape}")
            print(f"  Size: {size_mb:.2f} MB ({size_mb/file_size_mb*100:.1f}%)")
            total_analyzed += size_mb

        else:
            # For scalars and other types
            print(f"\n[{key}]")
            print(f"  Type: {type(value).__name__}")
            print(f"  Value: {value}")
            size_mb = get_size_mb(value)
            print(f"  Size: {size_mb:.4f} MB")
            total_analyzed += size_mb

    print(f"\n{'='*80}")
    print(f"Total analyzed: {total_analyzed:.2f} MB")
    print(f"File overhead: {file_size_mb - total_analyzed:.2f} MB")
    print(f"{'='*80}\n")


def compare_checkpoints(old_path, new_path):
    """Compare two checkpoint files"""
    print(f"\n{'='*80}")
    print("Comparing Checkpoints")
    print(f"{'='*80}")

    if not os.path.exists(old_path):
        print(f"Error: Old checkpoint not found: {old_path}")
        return
    if not os.path.exists(new_path):
        print(f"Error: New checkpoint not found: {new_path}")
        return

    old_size = os.path.getsize(old_path) / (1024 ** 2)
    new_size = os.path.getsize(new_path) / (1024 ** 2)
    size_diff = new_size - old_size

    print(f"\nOld checkpoint: {os.path.basename(old_path)}")
    print(f"  Size: {old_size:.2f} MB")

    print(f"\nNew checkpoint: {os.path.basename(new_path)}")
    print(f"  Size: {new_size:.2f} MB")

    print(f"\nDifference: {size_diff:+.2f} MB ({size_diff/old_size*100:+.1f}%)")

    # Load and compare keys
    old_ckpt = torch.load(old_path, map_location='cpu')
    new_ckpt = torch.load(new_path, map_location='cpu')

    old_keys = set(old_ckpt.keys())
    new_keys = set(new_ckpt.keys())

    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys

    if added_keys:
        print(f"\n✓ Added keys in new checkpoint:")
        for key in sorted(added_keys):
            value = new_ckpt[key]
            size_mb = get_size_mb(value)
            print(f"  - {key}: {type(value).__name__}, {size_mb:.4f} MB")

    if removed_keys:
        print(f"\n✗ Removed keys from old checkpoint:")
        for key in sorted(removed_keys):
            print(f"  - {key}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python check_checkpoint.py <checkpoint_path>")
        print("  python check_checkpoint.py <old_checkpoint> <new_checkpoint>")
        print("\nExample:")
        print("  python check_checkpoint.py core/data/exp/your_exp/ckpt/model_epoch_best.pth")
        print("  python check_checkpoint.py old_best.pth new_best.pth")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Analyze single checkpoint
        analyze_checkpoint(sys.argv[1])
    else:
        # Compare two checkpoints
        old_path = sys.argv[1]
        new_path = sys.argv[2]

        # First analyze both
        analyze_checkpoint(old_path)
        analyze_checkpoint(new_path)

        # Then compare
        compare_checkpoints(old_path, new_path)
