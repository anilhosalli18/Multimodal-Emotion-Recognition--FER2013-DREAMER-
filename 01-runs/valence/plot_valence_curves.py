import matplotlib.pyplot as plt


history = {
    "loss": [1.1005, 1.0934, 1.0916, 1.0872, 1.0898, 1.0843, 1.0785, 1.0779, 1.0752, 1.0775, 
             1.0701, 1.0653, 1.0650, 1.0652, 1.0654, 1.0602, 1.0606, 1.0584, 1.0553, 1.0547],
    "accuracy": [0.89, 0.892, 0.893, 0.891, 0.892, 0.893, 0.894, 0.895, 0.893, 0.892, 
                 0.894, 0.895, 0.896, 0.894, 0.891, 0.892, 0.893, 0.895, 0.894, 0.892],
    "val_loss": [1.1002, 1.0969, 1.0969, 1.1031, 1.0939, 1.1019, 1.0984, 1.0969, 1.1041, 1.1012, 
                 1.1062, 1.1114, 1.1057, 1.1086, 1.1089, 1.1089, 1.1069, 1.1014, 1.1031, 1.1138],
    "val_accuracy": [0.888, 0.890, 0.891, 0.889, 0.892, 0.892, 0.891, 0.890, 0.887, 0.893, 
                     0.891, 0.892, 0.892, 0.888, 0.887, 0.890, 0.891, 0.893, 0.889, 0.888]
}

# Extract metrics
train_acc = history['accuracy']
val_acc = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(train_acc)+1)

# Create figure
fig, ax1 = plt.subplots(figsize=(10,6))

# Plot Accuracy
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
ax1.plot(epochs, train_acc, 'o-', color='tab:blue', label='Training Accuracy')
ax1.plot(epochs, val_acc, 's-', color='tab:cyan', label='Validation Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim([0.85, 0.9])  # Zoom into 85%-90% for better clarity

# Plot Loss on right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red', fontsize=12)
ax2.plot(epochs, train_loss, 'o--', color='tab:red', label='Training Loss')
ax2.plot(epochs, val_loss, 's--', color='tab:orange', label='Validation Loss')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Combined legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.title('Training & Validation Accuracy and Loss', fontsize=14)
plt.grid(True)
plt.show()
