import os
import torch
import torch.nn as nn
from torch.nn import init
from transformers import BertModel
from .layers import CRF
from .layers.PositionAwareAttention import PositionAwareAttention
from torch.nn import CrossEntropyLoss

here = os.path.dirname(os.path.abspath(__file__))


class BertBilstmCrf(nn.Module):

    def __init__(self, hparams):
        super(BertBilstmCrf, self).__init__()
        self.pretrained_model_path = hparams.pretrained_model_path or 'Robert-base-chinese'
        self.embedding_dim = hparams.embedding_dim
        self.rnn_hidden_dim = hparams.rnn_hidden_dim
        self.rnn_num_layers = hparams.rnn_num_layers
        self.rnn_bidirectional = hparams.rnn_bidirectional
        # self.pe_dim=hparams.pe_dim
        # self.att_size=hparams.att_size
        self.max_len=hparams.max_len
        # self.constant_max_len=hparams.constant_max_len
        # self.dropout = hparams.dropout
        self.tagset_size = hparams.tagset_size
        # self.T_F_Att=hparams.T_F_Att

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)
        self.TF_ner = hparams.TF_ner
        self.ner2id = {'O': 0, 'ORGANIZATION': 1, 'LOCATION': 2, 'DATE': 3}
        self.ner_dim = hparams.ner_dim
        if self.TF_ner:
            self.ner_emb = nn.Embedding(len(self.ner2id), hparams.ner_dim,
                                        padding_idx=0)
        if self.TF_ner:
            self.lstm = nn.LSTM(self.embedding_dim+self.ner_dim,
                                self.rnn_hidden_dim // (2 if self.rnn_bidirectional else 1),
                                num_layers=self.rnn_num_layers, batch_first=True,
                                bidirectional=self.rnn_bidirectional)
        else:
            self.lstm = nn.LSTM(self.embedding_dim,
                                self.rnn_hidden_dim // (2 if self.rnn_bidirectional else 1),
                                num_layers=self.rnn_num_layers, batch_first=True,
                                bidirectional=self.rnn_bidirectional)
        self.hidden2tag = nn.Linear(self.rnn_hidden_dim, self.tagset_size)
        self.crf = CRF(num_tags=self.tagset_size)

    # def init_weights(self):
    #     # self.linear.bias.data.fill_(0)
    #     # init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
    #     if self.opt['attn']:
    #         self.pe_emb.weight.data.uniform_(-1.0, 1.0)

    def _get_emission_scores(self, input_ids, token_type_ids=None, attention_mask=None,ner_ids=None):
        embeds_bert = self.bert_model(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)[0]
        if self.TF_ner:
            embeds_ner=self.ner_emb(ner_ids)
            embeds = torch.cat([embeds_ner, embeds_bert], dim=-1)
        else:
            embeds=embeds_bert
        lstm_out, _= self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None,ner_ids=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask,ner_ids=ner_ids)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def decode(self, input_ids, token_type_ids=None, attention_mask=None,ner_ids=None):
        emissions = self._get_emission_scores(input_ids, token_type_ids, attention_mask,ner_ids=ner_ids)
        tags=self.crf.decode(emissions,mask=attention_mask.byte())
        return tags
