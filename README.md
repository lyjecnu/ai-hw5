# 第五次人工智能作业
# Repository Structure
├── data/                   #训练图像和文本
├── model_params/           #存储模型参数
│   └── __init__.py
├── training_stats/         #存储模型训练数据
│   ├── ablation.txt        #消融实验训练数据
│   ├── cross_attention.txt #交叉注意力模型训练数据
│   ├── fine_tuned_clip.txt #微调clip模型训练数据
│   ├── late_fusion.txt     #晚融合模型训练数据
│   └── simple_concat.txt   #简单拼接模型训练数据
├── ablation.py             #消融实验
├── cross_attention.py      #交叉注意力模型
├── fine_tuned_clip.py      #微调clip模型
├── late_fusion.py          #晚融合模型
├── load_data.py            #加载数据
├── requirements.txt        
├── simple_concat.py        #简单拼接模型
├── test_with_labels.txt    #有标签测试集结果
├── test_without_label.txt  #无标签测试集结果
├── train.txt               #训练标签
├── train_validate.py       #模型公用训练和验证函数
├── write_test.py           #写测试
└── zero_shot_clip_models.py   #零样本clip模型

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