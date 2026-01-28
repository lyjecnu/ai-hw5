# 第五次人工智能作业
# Repository Structure
```text
├── data/
├── model_params/
│   └── __init__.py
├── training_stats/
│   ├── ablation.txt
│   ├── cross_attention.txt
│   └── fine_tuned_clip.txt
├── ablation.py
├── cross_attention.py
├── requirements.txt
└── train_validate.py
```

# 准备工作
在data文件夹里放入图像和文字解释数据集

# 运行消融实验
python ablation.py

# 运行交叉注意力模型
python cross_attention.py

# 运行微调clip模型
python fine_tuend_clip.py

# 运行晚融合模型
python late_fusion.py

# 运行简单拼接模型
python simple_concat.py

# 运行零样本模型
python zero_shot_clip_models.py

# 调用模型并写入测试集标签
python write_test.py