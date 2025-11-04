"""
Additional layers.
"""
import torch
from torch import nn
import torch.nn.functional as F

class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """
    #input_size——hidden_dim
    #query_size——hidden_dim
    #feature_size——2*pe_dim pe_dim位置编码维度
    #attn_size——attn_dim attn_dim注意力大小

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        # self.init_weights()

    # def init_weights(self):
    #     self.ulinear.weight.data.normal_(std=0.001)
    #     ##改改
    #     # self.vlinear.weight.data.normal_(std=0.001)
    #     if self.wlinear is not None:
    #         self.wlinear.weight.data.normal_(std=0.001)
    #     self.tlinear.weight.data.zero_() # use zero to give uniform attention at the beginning

    #改改
    def forward(self, x, x_mask, q, f):
    # def forward(self, x, x_mask, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()
        # 看每个的shape变化
        ##改改
        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)#x [20,60,200]先平铺[1200,200]，再在通过线性变化[1200,200]，再变换维度成[20,60,200]
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(
                batch_size, seq_len, self.attn_size)#q[20,200]先平铺[20,200]，再在通过线性变化[20,200]，再扩张成[20,60,200]
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous().view(
                batch_size, seq_len, self.attn_size)#f[20,60,60]先平铺[1200,60]，再在通过线性变化[1200,200]，再变换维度成[20,60,200]
            # projs = [x_proj, f_proj]
            #改改
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(
            batch_size, seq_len)#[3,20,60,200]三个取和[20,60,200]，平铺[1200,200],线性变化[1200,1],再变换维度成[20,60]每个token对应一个attention值
        # d = queries.shape[-1]
        # # keys要做转置才能乘
        # scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # self.attention_weights = masked_softmax(scores, valid_lens)
        # self.attention_weights=weights.unsqueeze(-1)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))##masked为True的地方值变为-inf
        weights = F.sigmoid(scores)#att权值相加为1
        # weighted average input vectors
        weights = weights.unsqueeze(-1)
        print(weights)
        weights=weights.repeat(1,1,x.shape[-1])
        outputs = weights.mul(x)##bmm能够方便的实现三维数组的乘法 即weights在第二维度添加一个维度变为[20,1,60]后和x[20,60,200]相乘，变为[20,200]
        return outputs

