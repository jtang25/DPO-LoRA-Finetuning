import json
import matplotlib.pyplot as plt

# Load log
with open("training_log_qwen.json", "r") as f:
    log = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss
axes[0].plot(log["train_loss"], label="Train Loss", color="blue")
axes[0].set_xlabel("Step (x10)")
axes[0].set_ylabel("Loss")
axes[0].set_title("DPO Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot accuracy
axes[1].plot(log["train_acc"], label="Train Acc", color="blue")
# Eval acc is per-epoch, so plot at end of each epoch
if log["eval_acc"]:
    steps_per_epoch = len(log["train_acc"]) // len(log["eval_acc"])
    eval_x = [(i + 1) * steps_per_epoch - 1 for i in range(len(log["eval_acc"]))]
    axes[1].scatter(eval_x, log["eval_acc"], label="Eval Acc", color="red", s=100, zorder=5)
axes[1].axhline(y=0.5, color="gray", linestyle="--", label="Random Baseline")
axes[1].set_xlabel("Step (x10)")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("DPO Preference Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

print("Saved training_curves.png")
