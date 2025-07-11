import torch, transformers, cv2, wandb
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("OpenCV:", cv2.__version__)
print("W&B:", wandb.__version__)

wandb.init(project="emotion-text-gen")
wandb.log({"accuracy": 0.9})
wandb.finish()