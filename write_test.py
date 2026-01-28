import torch
from fine_tuned_clip import FineTunedCLIPModel  # 假设你的模型定义在此文件中
from load_data import get_clip_test_loader  # 确保该函数可导入

# 标签映射（与训练时一致）
idx_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}


def predict_test_set(model_path="model_params/fine_tuned_clip.pth", device=torch.device("cuda")):
    # 1. 加载模型
    model = FineTunedCLIPModel(num_classes=3, freeze_clip=True)
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()  # 切换到评估模式

    test_loader = get_clip_test_loader()

    guid_to_pred = {}

    #遍历测试集
    with torch.no_grad():
        for batch in test_loader:
            guids = batch['guid']
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # 前向传播
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

            # 获取预测类别（索引）
            preds = torch.argmax(logits, dim=1).cpu().tolist()  # 转为 list of int

            # 映射为标签并存入字典
            for guid, pred_idx in zip(guids, preds):
                guid_to_pred[guid] = idx_to_label[pred_idx]

    return guid_to_pred


# 使用示例
if __name__ == "__main__":
    predictions = predict_test_set()
    # 打印前5个结果示例
    for i, (guid, label) in enumerate(predictions.items()):
        if i >= 5:
            break
        print(f"{guid}: {label}")

    # 写入 test_with_labels.txt，格式为 guid,tag（例如 8,null）
    with open("test_with_labels.txt", "w") as f:
        for guid, label in predictions.items():
            # 如果 label 是 None，写成 "null"；否则写其字符串形式
            tag_str = "null" if label is None else str(label)
            f.write(f"{guid},{tag_str}\n")