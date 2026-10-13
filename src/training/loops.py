# Supervised and knowledge-distillation training loops.
import torch
import torch.nn.functional as F
from .optimization import create_optimizer, create_early_stopper, EarlyStopping, lr_lambda

def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataloader")
    loss = loss_total / total
    acc = correct / total

    return loss, acc

def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    early_stopper
):
    train_losses = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * labels.size(0)
            train_examples += labels.size(0)

        if train_examples == 0:
            raise ValueError("Cannot train on an empty dataloader")
        train_loss = train_loss_total / train_examples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}')

        # Cập nhật cả loss và acc vào early_stopper
        early_stopper(val_loss, val_acc, model)

        if early_stopper.early_stop:
            print("Dừng huấn luyện sớm!")
            break

        scheduler.step()

    return train_losses, val_losses, val_accs

def fit_kd(
    student, teacher, train_loader, val_loader, criterion,
    optimizer, scheduler, device, epochs, early_stopper,
    T=3.0, alpha=0.5
):
    """
    T: Temperature (Nhiệt độ) - giúp làm mềm xác suất.
    alpha: Trọng số cân bằng (alpha * KD_Loss + (1-alpha) * CE_Loss).
    """
    train_losses, val_losses, val_accs = [], [], []

    if T <= 0 or not 0 <= alpha <= 1:
        raise ValueError("T must be positive and alpha must be between 0 and 1")
    teacher.eval() # Teacher luôn ở chế độ eval

    for epoch in range(epochs):
        student.train()
        train_loss_total = 0.0
        train_examples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # 1. Forward pass
            student_logits = student(inputs)
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # 2. Soft Targets Loss (KL Divergence)
            # Ep student học theo phân phối của teacher
            soft_targets = F.softmax(teacher_logits / T, dim=1)
            soft_prob = F.log_softmax(student_logits / T, dim=1)
            distill_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (T**2)

            # 3. Hard Label Loss (Cross Entropy)
            student_loss = criterion(student_logits, labels)

            # 4. Total Loss
            loss = alpha * distill_loss + (1 - alpha) * student_loss

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * labels.size(0)
            train_examples += labels.size(0)

        # Đánh giá sau mỗi epoch
        if train_examples == 0:
            raise ValueError("Cannot train on an empty dataloader")
        train_loss = train_loss_total / train_examples
        val_loss, val_acc = evaluate(student, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Log và lưu checkpoint
        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}')
        early_stopper(val_loss, val_acc, student)

        if early_stopper.early_stop:
            print("Dừng huấn luyện sớm!")
            break

        scheduler.step()

    return train_losses, val_losses, val_accs
