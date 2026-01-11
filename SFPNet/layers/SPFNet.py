import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from SLIC.our_SLIC import spareaA
from torch_scatter.composite import scatter_softmax
from torch_scatter import scatter_add



class PFNet(nn.Module):
    def __init__(self, input_dim, output_dim, mask, head, dropout, use_dynamic_attention=None, ln=None, concat=True):
        super(PFNet, self).__init__()
        self.concat = concat
        self.head = head
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_dynamic_attention = use_dynamic_attention
        self.dropout = dropout
        self.ln = ln
        assert self.output_dim % self.head == 0, "output_dim must be divisible by head"
        self.output_dim_per_head = self.output_dim // self.head if self.concat else self.output_dim
        self.linvs = nn.ModuleList([
            nn.Linear(self.input_dim, self.output_dim_per_head)
            for _ in range(self.head)
        ])
        self.BN = nn.BatchNorm1d(self.output_dim)
        if self.use_dynamic_attention:
            self.linqs = nn.ModuleList([
                nn.Linear(self.input_dim, self.output_dim_per_head)
                for _ in range(self.head)
            ])
        if self.ln:
            self.ln = nn.LayerNorm(self.input_dim)
        self.src, self.dst, self.att_score = self.prefunction(mask)
        self.reset_parameters()

    def reset_parameters(self):
        for linv in self.linvs:
            linv.reset_parameters()
        self.BN.reset_parameters()
        if self.use_dynamic_attention:
            for linq in self.linqs:
                linq.reset_parameters()
        if self.ln:
            self.ln.reset_parameters()

    def forward(self, x):
        if self.ln:
            x = self.ln(x)
        v_list = [linv(x) for linv in self.linvs]
        att_scores = self.get_attscore(x)
        head_outputs = []
        for i in range(self.head):
            v_head = v_list[i]
            att_score_head = att_scores[:, i].view(-1, 1)
            agg = v_head[self.dst] * att_score_head
            out_head = scatter(agg, self.src, dim=0, dim_size=v_head.size(0), reduce='sum')
            head_outputs.append(out_head)
        if self.concat:
            out = torch.cat(head_outputs, dim=1)
        else:
            out = torch.stack(head_outputs, dim=0).mean(dim=0)
        out = self.BN(out)
        out = F.leaky_relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def prefunction(self, mask):
        mask_index = mask._indices()
        src, dst = mask_index[0], mask_index[1]
        att_score = mask.values()
        return src, dst, att_score.view(-1, 1)

    def get_attscore(self, x):
        if self.use_dynamic_attention:
            Q_list = [linq(x) for linq in self.linqs]
            head_att_scores = []
            for i in range(self.head):
                Q_head = Q_list[i]
                Q_head = F.layer_norm(Q_head, [Q_head.size(-1)])
                att_score_head = torch.sum((Q_head[self.src] * Q_head[self.dst]), dim=1) / self.output_dim
                att_score_head = scatter_softmax(att_score_head, self.src, dim=0)
                head_att_scores.append(att_score_head)
            att_scores = torch.stack(head_att_scores, dim=1)
            return att_scores
        else:
            return self.att_score.repeat(1, self.head)





class SFNet(MessagePassing):
    def __init__(self, input_dim, output_dim, A, dropout, ln=False, aggr='add'):
        super(SFNet, self).__init__(aggr=aggr)
        self.aggr = aggr
        self.lin = nn.Linear(input_dim, output_dim)
        self.BN = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.ln = ln
        if self.ln:
            self.ln_layer = nn.LayerNorm(input_dim)
        self.edge_index, self.edge_value, ssr = self.pre_get_adjceny(A)
        self.rhp, self.gama =5, 0.9
        self.reset_parameters()
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.BN.reset_parameters()
        if self.ln:
            self.ln_layer.reset_parameters()

    def forward(self, x):
        if self.ln:
            x = self.ln_layer(x)
        h = self.lin(x)
        x_start = self.propagate(edge_index=self.edge_index, x=h, edge_weight=self.edge_value)
        for i in range(self.rhp):
            x = (self.propagate(edge_index=self.edge_index, x=x, edge_weight=self.edge_value) + x_start)/ (2 + self.gama)
        output = self.BN(x)
        output = F.leaky_relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        return output

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def pre_get_adjceny(self, A):
        A, sparatio = spareaA(A, self.aggr)
        A = A.coalesce()
        edge_index = A._indices()
        edge_value = A.values()
        return edge_index, edge_value, sparatio



class Hyperpixel_SFNet(nn.Module):
    def __init__(self, input, output, Q, A,):
        super(Hyperpixel_SFNet, self).__init__()
        self.A = A
        self.norm_col_Q = self.normq(Q.T)
        self.Q = Q.to_sparse_csc()
        self.SFNet_Branch = nn.Sequential()
        for i in range(5):
            self.SFNet_Branch.add_module('SFNet_Branch' + str(i), SFNet(input_dim=input, output_dim=output, A=self.A, dropout=0.5))

    def forward(self, x):
        Hyperpixel = torch.sparse.mm(self.norm_col_Q, x)
        h = self.SFNet_Branch(Hyperpixel)
        result = torch.sparse.mm(self.Q, h)
        return result

    def normq(self, Q):
        Q = Q.to_sparse_coo().coalesce()
        index = Q.indices()
        value = Q.values()
        src, dst = index[0], index[1]
        row_sum = 1 / scatter_add(value, src, dim_size=Q.shape[0]).float()
        value = (row_sum[src]) * value
        Q = torch.sparse_coo_tensor(index, value, size=Q.size(), device=Q.device).coalesce()
        return Q.to_sparse_csc()


class SPFNet_A(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, mask,  q, a, hide):
        super(SPFNet_A, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        self.sfnet = Hyperpixel_SFNet(input=hide, output=hide, Q=q, A=a)
        self.pfnet = nn.Sequential()
        for i in range(2):
            self.pfnet.add_module('PFNet_Branch' + str(i), PFNet(input_dim=hide, output_dim=hide, mask=mask, head=2, dropout=0.5, use_dynamic_attention=True, ln=False))
        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 = self.pfnet(x) +  self.sfnet(x)
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y

class SPFNet_F(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, mask,  q, a, hide):
        super(SPFNet_F, self).__init__()
        self.class_count = class_count
        self.channel = changel
        self.height = height
        self.width = width
        self.BN = nn.BatchNorm1d(hide)
        self.prelin = nn.Linear(changel, hide)
        self.sfnet = Hyperpixel_SFNet(input=hide, output=hide, Q=q, A=a)
        self.pfnet = nn.Sequential()
        for i in range(2):
            self.pfnet.add_module('PFNet_Branch' + str(i), PFNet(input_dim=hide, output_dim=hide, mask=mask, head=2, dropout=0.5, use_dynamic_attention=False, ln=False))
        self.Softmax_linear = nn.Sequential(nn.Linear(hide, self.class_count))

    def forward(self, x: torch.Tensor):
        x = x.reshape([self.height * self.width, -1])
        x = (self.BN(self.prelin(x)))
        H1 = self.pfnet(x) +  self.sfnet(x)
        Y = self.Softmax_linear(H1)
        Y = F.softmax(Y, -1)
        return Y


























