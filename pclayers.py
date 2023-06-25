
# https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/drug_target_interaction/sign/layers.py
from pcutils import segment_pool

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM, Parameter
from torch.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax

from pcutils import generate_segment_id, get_index_from_counts


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.relu, bias=True):
        super(DenseLayer, self).__init__()
        self.activation = activation
        if not bias:
            self.fc = Linear(in_dim, out_dim, bias_attr=False)
        else:
            self.fc = Linear(in_dim, out_dim)
        self.reset_parameters()       

    def reset_parameters(self):
        self.fc.reset_parameters()     
    
    def forward(self, input_feat):
        return self.activation(self.fc(input_feat))


class SpatialInputLayer(nn.Module):
    """Implementation of Spatial Relation Embedding Module.
    """
    def __init__(self, hidden_dim, cut_dist, num_flt, activation= F.relu):
        super(SpatialInputLayer, self).__init__()
        self.cut_dist = cut_dist 
        self.num_flt = num_flt       
        self.dist_embedding_layer = nn.Embedding(int(cut_dist)-1, hidden_dim, sparse=False) 
        self.crt_input = nn.Linear(num_flt, hidden_dim)        
        self.softmax = nn.Softmax(dim=1)  
        self.dist_input_layer = DenseLayer(2*hidden_dim, hidden_dim, activation, bias=True)
        self.crt_input = DenseLayer(num_flt, hidden_dim, activation, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.crt_input.reset_parameters()   
        self.dist_input_layer.reset_parameters()
    
    def forward(self, edge_feat):
        dist   = torch.clip(edge_feat[:,0].squeeze(), 1.0, self.cut_dist-1e-6).long()-1
        eh_emb = self.dist_embedding_layer(dist) 

        ec_emb = self.crt_input(edge_feat[:,1:])        
        ec_emb = F.leaky_relu(ec_emb,.2) 
        ec_emb = self.softmax(ec_emb)
        eh_emb = torch.concat([eh_emb,ec_emb],dim = 1) 
        eh_emb = self.dist_input_layer(eh_emb)
        return eh_emb


class Atom2BondLayer(MessagePassing):
    """Implementation of Node->Edge Aggregation Layer.
    """
    def __init__(self, atom_dim, bond_dim, activation=F.relu):
        super(Atom2BondLayer, self).__init__(aggr='add')
        in_dim = atom_dim * 2 + bond_dim
        self.fc_agg = DenseLayer(in_dim, bond_dim, activation=activation, bias=True)
        #self.bond = nn.Linear(bond_dim,bond_dim)
           

    def forward(self, g, atom_h, bond):
        edge_index = g.edge_index        
        x = atom_h
        edge_attr = bond
        edge_attr = self.edge_updater(edge_index, x=x,edge_attr = edge_attr)
        return edge_attr

    def edge_update(self,x_j, x_i, edge_attr) -> Tensor: 
        #edge_attr = self.bond(edge_attr)       
        t = torch.cat([x_j, x_i, edge_attr],dim = 1)             
        return self.fc_agg(t)

    
class Bond2AtomLayer(MessagePassing):
    """Implementation of Distance-aware Edge->Node Aggregation Layer.
    """
    def __init__(self, bond_dim, atom_dim, 
                  hidden_dim, num_heads, 
                  dropout, merge='mean', activation=F.relu,
                  **kwargs,):
        kwargs.setdefault('aggr', 'add')           
        super(Bond2AtomLayer, self).__init__(node_dim=0, **kwargs)
        self.merge = merge
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.src_fc = Linear(bond_dim, hidden_dim)
        self.dst_fc = Linear(atom_dim, hidden_dim)
        self.edg_fc = Linear(hidden_dim, hidden_dim)
        
        self.src_fc = Linear(bond_dim, num_heads * hidden_dim)
        self.dst_fc = Linear(atom_dim, num_heads * hidden_dim)
        self.edg_fc = Linear(hidden_dim, num_heads * hidden_dim)    

        self.attn_src  = Parameter(torch.Tensor(1,num_heads, hidden_dim))        
        self.attn_bond = Parameter(torch.Tensor(1,num_heads, hidden_dim))
        self.attn_edge = Parameter(torch.Tensor(1,num_heads, hidden_dim))
        
        
        self.drop = dropout        
        self.negative_slope = 0.2        
        self.activation = activation   
        self.tanh = nn.Tanh()  
       # self.softmax = nn.Softmax(dim=1)    
        self.reset_parameters()
        
    def reset_parameters(self):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters() 
        self.edg_fc.reset_parameters()
        
        glorot(self.attn_src)       
        glorot(self.attn_bond)
        glorot(self.attn_edge)


    def forward(self, g, x, bond_feat,edge_attr):              
        edge_index = g.edge_index           
        x          = F.dropout(x, p=self.drop, training=self.training)  
        x =  self.dst_fc(x).view(-1, self.num_heads, self.hidden_dim)        
        alpha = (x * self.attn_src).sum(dim=-1)  
        bond_feat = F.dropout(bond_feat,p=self.drop, training=self.training)
        edge_attr = F.dropout(edge_attr,p=self.drop, training=self.training)        
        bond_feat = self.src_fc(bond_feat).view(-1, self.num_heads, self.hidden_dim)
        alpha = self.edge_updater(edge_index, alpha = alpha, bond_feat = bond_feat, edge_attr = edge_attr)

        out = self.propagate(edge_index,
                             bond_feat = bond_feat, 
                             alpha = alpha)  
                          
        if self.merge == 'cat':            
            out = out.view(-1, self.num_heads * self.hidden_dim)              
        if self.merge == 'mean':            
            out = torch.mean(out,dim = 1)                                    
        if self.activation:
            rst = self.activation(out) 
        return rst    

    def edge_update(self,  alpha_j, bond_feat, edge_attr,index,ptr,size_i): 
        edge_attr = self.edg_fc(edge_attr).view(-1, self.num_heads, self.hidden_dim)
        alpha_edge  = (edge_attr * self.attn_edge).sum(dim=-1)
        alpha_bond  = (bond_feat * self.attn_bond).sum(dim=-1)        
        alpha = alpha_j + alpha_edge + alpha_bond
        #alpha = F.leaky_relu(alpha, self.negative_slope)
        #alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.tanh(alpha)
        alpha = F.dropout(alpha, p=self.drop, training=self.training)          
        return alpha

    def message(self, bond_feat, alpha ): 
        return bond_feat* alpha.unsqueeze(-1)
    
        
class DomainAttentionLayer(MessagePassing):
    """Implementation of Angle Domain-speicific Attention Layer.
    """
    def __init__(self, bond_dim, hidden_dim, dropout, activation=F.relu):
        super(DomainAttentionLayer, self).__init__(aggr = 'add')
        self.attn_fc = Linear(2 * bond_dim, hidden_dim)
        #self.attn_out = nn.Linear(hidden_dim, 1)
        self.drop = dropout 
        self.tanh = nn.Tanh()
        self.activation = activation  
        self.reset_parameters()

    def reset_parameters(self):
        self.attn_fc.reset_parameters() 
    

    def forward(self, g, bond_feat):
        edge_index = g.edge_index
        edge_index = edge_index.long()        
        bond_feat = F.dropout(bond_feat, p = self.drop, training=self.training)  
        rst = self.propagate(edge_index, x = bond_feat )
           
        if self.activation:
            rst = self.activation(rst)
        return rst

    def message(self, x_i, x_j,index,ptr,size_i) -> Tensor:
        alpha = torch.concat([x_i,x_j],dim =-1)
        alpha = self.attn_fc(alpha)        
        alpha = self.tanh(alpha)  
        #alpha = softmax(alpha, index, ptr, size_i)              
        alpha = F.dropout(alpha, p = self.drop, training=self.training) 
        #alpha = self.attn_out(alpha)           
        return alpha*x_j 

    
     

class Bond2BondLayer(nn.Module):
    """Implementation of Angle-oriented Edge->Edge Aggregation Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle, dropout, merge='cat', activation=None):
        super(Bond2BondLayer, self).__init__()
        self.num_angle = num_angle
        self.hidden_dim = hidden_dim
        self.merge = merge
        self.conv_layer = nn.ModuleList()
        for _ in range(num_angle):
            conv = DomainAttentionLayer(bond_dim, hidden_dim, dropout, activation=None)
            self.conv_layer.append(conv)
        self.activation = activation
    
    def forward(self, g_list, bond_feat):
       
        h_list = []
        for k in range(self.num_angle):
            h = self.conv_layer[k](g_list[k], bond_feat)
            h_list.append(h)

        if self.merge == 'cat':
            feat_h = torch.cat(h_list, dim=1)
        if self.merge == 'mean':
            feat_h = torch.mean(torch.stack(h_list, dim=-1), dim=1)
        if self.merge == 'sum':
            feat_h = torch.sum(torch.stack(h_list, dim=-1), dim=1)
        if self.merge == 'max':
            feat_h = torch.max(torch.stack(h_list, dim=-1), dim=1)
        if self.merge == 'cat_max':
            feat_h = torch.stack(h_list, dim=-1)
            feat_max = torch.max(feat_h, dim=1)[0]
            feat_max = torch.reshape(feat_max, [-1, 1, self.hidden_dim])
            feat_h = torch.reshape(feat_h * feat_max, [-1, self.num_angle * self.hidden_dim])

        if self.activation:
            feat_h = self.activation(feat_h)           
        return feat_h
    

class PiPoolLayer(nn.Module):
    """Implementation of Pairwise Interactive Pooling Layer.
    """
    def __init__(self, bond_dim, hidden_dim, num_angle):
        super(PiPoolLayer, self).__init__()
        self.bond_dim = bond_dim
        self.num_angle = num_angle
        self.num_type = 4 * 9
        fc_in_dim = num_angle * bond_dim
        self.fc_1 = DenseLayer(fc_in_dim, hidden_dim, activation=F.relu)
        self.fc_2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, bond_types_batch, type_count_batch, bond_feat):
        """
        Input example:
            bond_types_batch: 边类型 #[batch_size 个图边的数量和]
            type_count_batch: 各类型边的数量# [num_type*batch_size]
        """
        #调整 type_count_batch 为 [num_type, batch_size]
        type_count_batch =  type_count_batch.reshape(-1,self.num_type).transpose(0,1) 
       
        bond_feat = self.fc_1(torch.reshape(bond_feat, [-1, self.num_angle*self.bond_dim]))
        inter_mat_list =[]
        for type_i in range(self.num_type):
            type_i_index = torch.masked_select(
                torch.arange(bond_feat.shape[0], device=bond_types_batch.device), 
                bond_types_batch==type_i
            )
            if torch.sum(type_count_batch[type_i]) == 0:
                inter_mat_list.append(torch.zeros(type_count_batch[type_i].shape[0], 
                                                   dtype=torch.float, 
                                                   device=bond_types_batch.device)
                                                   )
                continue

            bond_feat_type_i = bond_feat[type_i_index,:]
            #如果batch_size = 3 个图， 每个图的type i 边 分别有[2,3,4]，一共9个边如果放在一个向量里面，
            #得到9维向量，前两个是[1,1,2,2,2,3,3,3,3], 对每个图的type i边池化，只需要区别池化然后合并
            # 结果。

            #下面即是给出各个图的type i 边的数量，然后标志图的边的七点位置例如[0,2,5,9],最后一个数字是总边数.
            graph_bond_index = get_index_from_counts(type_count_batch[type_i],bond_types_batch.device)
            
            #设置一个9维零向量，在起点位置取1，[1,0,1,0,0,1,0,0,0], 通过累加得到各个type i边的位置
            # [1,1,2,2,2,3,3,3,3]
            graph_bond_id = generate_segment_id(graph_bond_index)    
            # bond_feat_type_i 也是 9维向量，比照 [1,1,2,2,2,3,3,3,3] 的位置，相同数字的位置求和，
            # 三个和并成一个向量，即是各个图type i边的池化。      
            graph_feat_type_i = segment_pool(bond_feat_type_i, graph_bond_id)
            mat_flat_type_i = self.fc_2(graph_feat_type_i).squeeze(1)
            my_pad = nn.ConstantPad1d((0,len(type_count_batch[type_i])-len(mat_flat_type_i)), -1e9)
            mat_flat_type_i = my_pad(mat_flat_type_i)
            inter_mat_list.append(mat_flat_type_i)

        inter_mat_batch = torch.stack(inter_mat_list, dim=1) # [batch_size, num_type]        
        inter_mat_mask = torch.ones_like(inter_mat_batch) * -1e9
        inter_mat_batch = torch.where(type_count_batch.transpose(1, 0)>0, inter_mat_batch, inter_mat_mask)
        inter_mat_batch = self.softmax(inter_mat_batch)
        return inter_mat_batch

from torch_geometric.nn import global_add_pool
class OutputLayer(nn.Module):
    """Implementation of Prediction Layer.
    """
    def __init__(self, atom_dim, hidden_dim_list):
        super(OutputLayer, self).__init__()
        self.pool = global_add_pool
        self.mlp = nn.ModuleList()
        for hidden_dim in hidden_dim_list:
            self.mlp.append(DenseLayer(atom_dim, hidden_dim, activation=F.relu))
            atom_dim = hidden_dim
        self.output_layer = nn.Linear(atom_dim, 1)
    
    def forward(self, atom_feat,a2a_g):
        graph_feat = self.pool(atom_feat, a2a_g.batch)
        for layer in self.mlp:
            graph_feat = layer(graph_feat)
        output = self.output_layer(graph_feat)
        return output

    