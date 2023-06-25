# Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/drug_target_interaction/sign/model.py
"""
Model code for Curvature-based Adapative Graph Neural Networks (CurvAGN).

"""

import torch.nn as nn
import torch.nn.functional as F
from pclayers import SpatialInputLayer, Atom2BondLayer, Bond2BondLayer, Bond2AtomLayer, PiPoolLayer, OutputLayer

class SIGN(nn.Module):
    def __init__(self, args):
        super(SIGN, self).__init__()
        num_convs = args.num_convs
        dense_dims = args.dense_dims
        infeat_dim = args.infeat_dim
        hidden_dim = args.hidden_dim
        self.num_convs = num_convs

        cut_dist = args.cut_dist
        num_flt = args.num_flt

        num_angle = args.num_angle
        merge_b2b = args.merge_b2b
        merge_b2a = args.merge_b2a

        activation = args.activation
        num_heads = args.num_heads
        feat_drop = args.feat_drop

        self.input_layer = SpatialInputLayer(hidden_dim, cut_dist, num_flt, activation=F.relu)
        self.atom2bond_layers = nn.ModuleList()
        self.bond2bond_layers = nn.ModuleList()
        self.bond2atom_layers = nn.ModuleList()
        for i in range(num_convs):
            if i == 0:
                atom_dim = infeat_dim
            else:
                atom_dim = hidden_dim * num_heads if 'cat' in merge_b2a else hidden_dim
            bond_dim = hidden_dim * num_angle if 'cat' in merge_b2b else hidden_dim

            self.atom2bond_layers.append(Atom2BondLayer(atom_dim, bond_dim=hidden_dim, activation=activation))
            self.bond2bond_layers.append(Bond2BondLayer(hidden_dim, hidden_dim, num_angle, feat_drop, merge=merge_b2b, activation=None))
            self.bond2atom_layers.append(Bond2AtomLayer(bond_dim,atom_dim, hidden_dim, num_heads, feat_drop,  merge=merge_b2a, activation=activation))
            
        self.pipool_layer = PiPoolLayer(hidden_dim, hidden_dim, num_angle)
        self.output_layer = OutputLayer(hidden_dim, dense_dims)
        
    
    def forward(self, a2a_g, b2a_g, b2b_gl):
        
        atom_h = a2a_g.x
        dist_feat =  a2a_g.edge_attr
        bond_types = a2a_g.bond_types
        type_count = a2a_g.type_count
        dist_h = self.input_layer(dist_feat)

        for i in range(self.num_convs):
            bond_h = self.atom2bond_layers[i](a2a_g, atom_h, dist_h)
            bond_h = self.bond2bond_layers[i](b2b_gl, bond_h)           
            atom_h = self.bond2atom_layers[i](b2a_g, atom_h, bond_h, dist_h)

        pred_inter_mat = self.pipool_layer(bond_types, type_count, bond_h)
        pred_socre = self.output_layer(atom_h,a2a_g)
        return pred_inter_mat, pred_socre

