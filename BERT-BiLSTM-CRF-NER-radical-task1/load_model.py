from transformers import BertConfig, BertModel
import os

pretrained_path = "./radical_bert_model/"
config_path = os.path.join(pretrained_path, "config.json")
checkpoint_path = os.path.join(pretrained_path, "model.ckpt-20")
vocab_path = os.path.join(pretrained_path, 'vocab.txt')
model_bin_path=os.path.join(pretrained_path, 'pytorch_model.bin')

# 加载config
config = BertConfig.from_json_file(config_path)
# 加载原始模型
tfbert_model1 = BertModel.from_pretrained(pretrained_model_name_or_path=model_bin_path, config=config)
# # 加载分类模型
# tfbert_model2 = TFBertForSequenceClassification.from_pretrained(pretrained_path, from_pt=True, config=config)
