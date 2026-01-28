import torch
import torch.nn as nn

def train_clip(
    model,
    optimizer,
    dataloader,
    num_epochs=5,
    device=None,
    log_interval=100,
    save_path=None
):
    """
    通用训练函数，适用于 FineTunedCLIPModel 或类似结构。
    要求 dataloader 的 batch 包含以下字段：
        - 'pixel_values'   (processed image tensor)
        - 'input_ids'
        - 'attention_mask'
        - 'label'

    Args:
        model (nn.Module): 待训练模型（如 FineTunedCLIPModel）
        optimizer (torch.optim.Optimizer): 优化器
        dataloader (DataLoader): 训练数据加载器
        num_epochs (int): 训练轮数
        device (torch.device or str): 设备（如 'cuda' 或 'cpu'）
        log_interval (int): 每隔多少 batch 打印一次日志
        save_path (str, optional): 模型保存路径
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):

            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # 前向传播（适配 FineTunedCLIPModel）
            logits = model(pixel_values, input_ids, attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"[Epoch {epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # 保存模型
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

def train_model(
    model,
    optimizer,
    dataloader,
    num_epochs=5,
    device=None,
    log_interval=100,
    save_path=None
):
    """
    通用训练函数，适用于任何返回 logits 的 PyTorch 模型，
    要求 dataloader 的 batch 包含以下字段：
        - 'image'
        - 'input_ids'
        - 'attention_mask'
        - 'label'

    Args:
        model (nn.Module): 待训练模型
        optimizer (torch.optim.Optimizer): 优化器
        dataloader (DataLoader): 训练数据加载器
        num_epochs (int): 训练轮数
        device (torch.device or str): 设备（如 'cuda' 或 'cpu'）
        log_interval (int): 每隔多少 batch 打印一次日志
        save_path (str, optional): 模型保存路径（训练结束后自动保存）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(dataloader):
            # 将必要字段移到设备上
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # 前向：支持任意模型，只要它接受这三个输入
            logits = model(images, input_ids, attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        print(f"[Epoch {epoch+1}/{num_epochs}] Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

    # 保存模型
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")



def validate(model, val_loader, device='cpu'):
    """
    在验证集上评估模型性能。

    Args:
        model (torch.nn.Module): 要评估的模型。
        val_loader (DataLoader): 验证数据加载器。
        device (str): 运行设备，如 'cpu' 或 'cuda'。

    Returns:
        tuple: (accuracy, avg_loss)
            - accuracy (float): 验证集准确率（0～1）
            - avg_loss (float): 验证集平均损失
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    # 使用 CrossEntropyLoss（即使模型不返回 loss）
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():  # 关闭梯度计算
        for batch in val_loader:
            # 将数据移到指定设备
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits = model(
                image=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 手动计算 loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)



            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return accuracy, avg_loss



def clip_validate(model,val_loader,device):
    """
    在验证集上评估模型性能。

    Args:
        model (torch.nn.Module): 要评估的模型。
        val_loader (DataLoader): 验证数据加载器。
        device (str): 运行设备，如 'cpu' 或 'cuda'。

    Returns:
        tuple: (accuracy, avg_loss)
            - accuracy (float): 验证集准确率（0～1）
            - avg_loss (float): 验证集平均损失
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    # 使用 CrossEntropyLoss（即使模型不返回 loss）
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():  # 关闭梯度计算
        for batch in val_loader:
            # 将数据移到指定设备
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 手动计算 loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)



            total_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return accuracy, avg_loss
