from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn
from model.graph_encoder import truncated_normal
torch.set_printoptions(precision=16)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class HGNNLayer(torch.nn.Module):
    def __init__(self,
                 adj,
                 input_dim,
                 output_dim,
                 bias=True,
                 act=None):
        super(HGNNLayer,self).__init__()
        self.bias = bias
        self.act = act
        self.adj = adj
        self.weight_mat = torch.nn.parameter.Parameter(torch.zeros([input_dim, output_dim]))
        self.weight_mat.data=truncated_normal(self.weight_mat.data,std=0.02)

        if bias:
            self.bias_vec = torch.nn.init.constant_(torch.nn.parameter.Parameter(torch.zeros([1, output_dim])),0.0)

    def forward(self, inputs, drop_rate=0.0):
        pre_sup_tangent = inputs
        if drop_rate > 0.0:
            pre_sup_tangent = torch.nn.Dropout(p=drop_rate)(pre_sup_tangent) * (1 - drop_rate)  # not scaled up
        output = torch.matmul(pre_sup_tangent, self.weight_mat)
        output = torch.sparse.mm(self.adj, output)
        if self.bias:
            bias_vec = self.bias_vec
            output = torch.add(output, bias_vec) 
        if self.act is not None:
            output = self.act(output)
        return output