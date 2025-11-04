import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
import torch
from d2l import torch as d2l
import torch.nn.functional as F
import math
from torch.autograd import Variable
'''
class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert0 = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None):
        outputs = self.bert0(input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
        # print("outputs_bert："+str(np.array(outputs).shape))
        # print(outputs)
        pooled_output = outputs[1]
        # print("outputs_bert2：" + str(pooled_output.detach().numpy().shape))
        pooled_output = self.dropout(pooled_output)
        # print("outputs_dropout：" + str(pooled_output.detach().numpy().shape))
        logits = self.classifier(pooled_output)
        # print("logits："+str(logits.detach().numpy().shape))
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        # You can unfreeze the last layer of bert0 by calling set_trainable(model.bert0.encoder.layer[23], True)
        set_trainable(self.bert0, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert0.encoder.layer[i], True)
'''

class BertForMultiLable(BertPreTrainedModel):

    def __init__(self, config, need_birnn=True, rnn_dim=128):
        super(BertForMultiLable, self).__init__(config)

        self.num_tags = config.num_labels
        self.bert = BertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.need_birnn = need_birnn

        # 如果为False，则不要BiLSTM层
        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2
        self.classifier = nn.Linear(out_dim, config.num_labels)#nn.Linear(config.hidden_size, config.num_labels)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    # @save
    def masked_softmax(self,X, valid_lens):
        """通过在最后一个轴上掩蔽元素来执行softmax操作"""
        # X:3D张量，valid_lens:1D或2D张量
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
            X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                                  value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)

    def forward(self, input_ids, token_type_ids=None, input_mask=None,head_mask=None,triggers=None):#self, input_ids, token_type_ids=None, attention_mask=None,head_mask=None
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask,head_mask=head_mask)
        sequence_output = outputs[0]#output[1]是池化之后的2维，output[0]是池化之前的3维
        # sequence_output = self.dropout(sequence_output)
        if self.need_birnn:
            sequence_output,_= self.birnn(sequence_output)

        # sequence_output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        # max_len=input_mask.shape[1]
        # T_F=[False]*max_len
        # for trigger in triggers:
        #     for this_index in range(trigger[0],trigger[1]):
        # print(sequence_output)

        query = torch.zeros(sequence_output.shape[0],1,sequence_output.shape[2])
        for trigger_index,trigger in enumerate(triggers):
            query[trigger_index][0]=torch.sum(sequence_output[trigger_index][trigger[0]:trigger[1]], dim=0)
        # print(query)
        d = query.shape[-1]
        # keys要做转置才能乘
        scores = torch.bmm(query, sequence_output.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens=None)
        final_output=torch.bmm(self.attention_weights, sequence_output)
        final_output=torch.squeeze(final_output, dim=1)

        logits = self.classifier(final_output)
        # logits=self.softmax(logits)
        # outputs=F.softmax(logits.view(sequence_output.shape[0], -1))
        # logits = self.dropout(logits)#把dropout层从加在BERT层改为加在全连接层，可以多加几个试试
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))
        # You can unfreeze the last layer of bert0 by calling set_trainable(model.bert0.encoder.layer[23], True)
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer + 1):
            set_trainable(self.bert.encoder.layer[i], True)
