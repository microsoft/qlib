import sys
import pickle
import torch
import os

def inspect_params(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"Pickle load failed: {e}")
        try:
            import dill
            print("Trying dill...")
            with open(file_path, "rb") as f:
                obj = dill.load(f)
        except ImportError:
            print("Dill not installed, and pickle failed.")
            return
        except Exception as e:
            print(f"Dill load failed: {e}")
            return

    print(f"\nType: {type(obj)}")
    
    # Print Qlib Model Hyperparameters
    print("\n=== Model Hyperparameters ===")
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            # Skip private attributes and large objects
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, str, bool)):
                print(f"{k}: {v}")
            elif isinstance(v, (list, tuple)) and len(v) < 20:
                print(f"{k}: {v}")
            elif v is None:
                print(f"{k}: None")
            else:
                # For other objects, just print type to avoid clutter
                if k not in ["model", "net"]:
                    print(f"{k}: <{type(v).__name__}>")

    # Inspect PyTorch Structure
    print("\n=== PyTorch Architecture ===")
    torch_model = None
    if hasattr(obj, "model") and isinstance(obj.model, torch.nn.Module):
        torch_model = obj.model
    elif hasattr(obj, "net") and isinstance(obj.net, torch.nn.Module):
        torch_model = obj.net
    
    if torch_model:
        print(torch_model)
        
        # Print parameter count
        total_params = sum(p.numel() for p in torch_model.parameters())
        trainable_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
    else:
        print("No PyTorch model found in 'model' or 'net' attributes.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_params.pkl>")
    else:
        inspect_params(sys.argv[1])
