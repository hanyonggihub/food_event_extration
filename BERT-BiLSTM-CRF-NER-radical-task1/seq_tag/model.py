import os
import torch
import torch.nn as nn
from transformers import BertModel,BertConfig
from .layers import CRF
from torch.nn import CrossEntropyLoss

here = os.path.dirname(os.path.abspath(__file__))


class BertBilstmCrf(nn.Module):

    def __init__(self, hparams):
        super(BertBilstmCrf, self).__init__()

        self.pretrained_model_path = hparams.pretrained_model_path or 'bert-base-chinese'
        self.pretrained_radical_model_path=hparams.radical_model_file
        self.pretrained_radical_model_config_path = hparams.radical_model_config_file
        self.embedding_dim = hparams.embedding_dim
        self.radical_embedding_dim = hparams.radical_embedding_dim
        # self.rnn_hidden_dim = hparams.rnn_hidden_dim
        # self.rnn_num_layers = hparams.rnn_num_layers
        # self.rnn_bidirectional = hparams.rnn_bidirectional
        # self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)
        config = BertConfig.from_json_file(self.pretrained_radical_model_config_path)
        # 加载原始模型
        self.radical_bert_model =BertModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_radical_model_path, config=config)
        # self.lstm = nn.LSTM(self.embedding_dim+self.radical_embedding_dim,
        #                     self.rnn_hidden_dim // (2 if self.rnn_bidirectional else 1),
        #                     num_layers=self.rnn_num_layers, batch_first=True,
        #                     bidirectional=self.rnn_bidirectional)
        # self.drop = nn.Dropout(self.dropout)
        # self.rnn_hidden_dim需要改一下
        # self.hidden2tag = nn.Linear(self.rnn_hidden_dim, self.tagset_size)
        # self.hidden2tag = nn.Linear(self.embedding_dim + self.radical_embedding_dim, self.tagset_size)
        self.hidden2tag = nn.Linear(self.embedding_dim, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size)


    def _get_emission_scores(self, input_ids, token_type_ids=None, attention_mask=None,radical_input_ids=None, radical_token_type_ids=None, radical_attention_mask=None):
        char_embeds = self.bert_model(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)[0]
        ##!!!!!!#input_ids是不一样的
        ##改改还有问题
        # radical_embeds=self.radical_bert_model(radical_input_ids, attention_mask=radical_attention_mask, token_type_ids=radical_token_type_ids)[0]##token_type_ids=token_type_ids #token_type_ids=segment_ids
        #参数type_id需要有吗？看训练代码去
        # embeds=torch.cat([radical_embeds,char_embeds],dim=-1)
        #mask不变！
        # lstm_out, _ = self.lstm(embeds)
        # lstm_dropout = self.drop(lstm_out)
        # emissions = self.hidden2tag(lstm_dropout)
        emissions = self.hidden2tag(char_embeds)
        return emissions

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None,radical_input_ids=None, radical_token_type_ids=None, radical_attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask,radical_input_ids, radical_token_type_ids, radical_attention_mask)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None,radical_input_ids=None, radical_token_type_ids=None, radical_attention_mask=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask,radical_input_ids, radical_token_type_ids, radical_attention_mask)
        #改decode
        #softmax得到标签
        #emissions_socre=nn.softmax(emissions,dim=1)
        tags=self.crf.decode(emissions,mask=attention_mask.byte())
        return tags
