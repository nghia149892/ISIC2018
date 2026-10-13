# Optional visualizations and prediction collection.
import torch
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .reproducibility import set_seed

def show_train_sample(train_loader, classes):

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    img = images[0] * std + mean

    plt.imshow(transforms.ToPILImage()(img))
    plt.title(classes[labels[0].item()])
    plt.axis("off")
    plt.show()

def show_validation_samples(val_loader, classes, num_images=5):

    images, labels = next(iter(val_loader))

    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

    plt.figure(figsize=(15,3))

    for i in range(num_images):

        img = images[i] * std + mean
        img = img.numpy().transpose((1,2,0))
        img = img.clip(0,1)

        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(classes[labels[i].item()])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_loss_acc(train_losses, val_losses, val_accs):
    plt.figure(figsize=(18, 5))

    # ================= Train Loss =================
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, color='green', linewidth=2)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    # ================= Validation Loss =================
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, color='blue', linewidth=2)
    plt.title("Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.6)

    # ================= Validation Accuracy =================
    plt.subplot(1, 3, 3)
    plt.plot(val_accs, color='orange', linewidth=2)
    plt.title("Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def get_all_predictions(model, dataloader, device):
    """Hàm lấy toàn bộ dự đoán và nhãn thực tế"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

def plot_confusion_matrix(all_labels, all_preds, label_names, figsize=(8,6)):
    """Hàm vẽ confusion matrix"""
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(label_names))),
    )
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
