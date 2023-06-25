# Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/drug_target_interaction/sign/dataset.py
"""
Dataset code for protein-ligand complexe interaction graph construction.
"""

import os
from GraphRicciCurvature.FormanRicci import FormanRicci
import networkx as nx
import numpy as np
from pcutils import cos_formula

import torch

from torch_geometric.data import Data
import pickle
from torch_geometric.data import  Dataset
from torch_geometric.loader import DataListLoader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from tqdm import tqdm

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]

class ComplexDataset(Dataset):
    def __init__(self, root, data_dir, dataset, cut_dist, num_angle, num_flt, save_file=True,
     test=False, val = False, transform=None, pre_transform=None):
        
        self.data_path = root
        self.data_dir = data_dir 
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.num_angle = num_angle
        self.num_flt = num_flt
        self.save_file = save_file
    
        self.labels = []
        self.a2a_graphs = []
        self.b2a_graphs = []
        self.b2b_grpahs_list = []
        self.inter_feats_list = []
        self.bond_types_list = []
        self.type_count_list = []

        self.test = test
        self.val = val
        #self.index = 0
        self.filename = os.path.join(data_dir, "{0}.pkl".format(dataset))
        
        super(ComplexDataset, self).__init__(root, transform, pre_transform)
       
                               
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        with open(self.raw_paths[0], 'rb') as f:
            self.data = pickle.load(f)

        
        if self.test:
            return [f'data_test_{i}.pt' for i in range(len(self.data[1]))]
        elif self.val:  
            return  [f'data_val_{i}.pt' for i in range(len(self.data[1]))] 
        else:
            return [f'data_train_{i}.pt' for i in range(len(self.data[1]))]   

    
    
    def _build_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol
        
        #print('d',inter_feats)

        ##################################################
        # prepare distance matrix and interaction matrix #
        ##################################################
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        inter_feats = np.array([inter_feats])
        inter_feats = inter_feats / inter_feats.sum()

        ############################
        # build atom to atom graph #
        ############################
        num_atoms = len(coords)
        dist_graph_base = dist_mat.copy()
        dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1,1)
        
        t = dist_feat.shape[0] #wu
        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)              
        a2a_edges = list(zip(atom_graph.row, atom_graph.col))                   
        wh = np.where(dist_graph_base > 0)
        # crt = torch.zeros((t,self.num_flt))
        # adj = atom_graph.toarray()        
        # g_x = nx.Graph(adj) 
        # for (n1, n2, d) in g_x.edges(data=True):
        #     d.clear()
        # orc = FormanRicci(g_x, verbose="TRACE")
        # orc.compute_ricci_curvature()
        # G_orc = orc.G.copy() 
        # for k in range(t): #wu
        #     i = wh[0][k]
        #     j = wh[1][k]
        #     if dist_graph_base[i,j] == 0:
        #         crt[k,0] = 0
        #     elif  j > i:  
        #         crt[k,0] = G_orc[i][j]['formanCurvature']                    
        #     else:    
        #         crt[k,0] = G_orc[j][i]['formanCurvature']
                    
       
            
        
        
        crt = torch.zeros((t,self.num_flt))
        
        for kk in  range(self.num_flt):
            tmp_dist = dist_graph_base.copy()           
            ds = self.cut_dist/self.num_flt
            ds_flt = kk*ds
            
            tmp_dist[tmp_dist > ds_flt] = 0#这样操作实际上没有考虑曲率, 应该由倒过来.
            
                      
            ag = coo_matrix(tmp_dist)
            adj = ag.toarray()
            g_x = nx.Graph(adj)
            orc = FormanRicci(g_x,  verbose="TRACE")
            orc.compute_ricci_curvature()
            G_orc = orc.G.copy() 
            
            for k in range(t): #wu
                i = wh[0][k]
                j = wh[1][k]
                if tmp_dist[i,j] == 0:
                    crt[k,kk] = 0
                elif  j > i:  
                    crt[k,kk] = G_orc[i][j]['formanCurvature']                    
                else:    
                    crt[k,kk] = G_orc[j][i]['formanCurvature']
                    
        dist_feat = torch.tensor(dist_feat, dtype=torch.float)    
        
        edge_ft = torch.hstack((dist_feat,crt))
        features = torch.tensor(features,dtype=torch.float)
        a2a_edges = torch.tensor(a2a_edges,dtype=torch.long).t().contiguous()
        num_atoms = torch.tensor(num_atoms, dtype = torch.long)
       
        
        a2a_graph = Data(x = features, edge_index = a2a_edges, num_nodes=num_atoms,  edge_attr= edge_ft)

        ######################
        # prepare bond nodes #
        ######################
        indices = []
        bond_pair_atom_types = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                a = dist_mat[i, j]
                if a < self.cut_dist:
                    at_i, at_j = atoms[i], atoms[j]                    
                    if i < num_atoms_d and j >= num_atoms_d and (at_j, at_i) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_j, at_i))]
                    elif i >= num_atoms_d and j < num_atoms_d and (at_i, at_j) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_i, at_j))]
                    else:
                        bond_pair_atom_types += [-1]
                    indices.append([i, j])

        ############################
        # build bond to atom graph #
        ############################
        num_bonds = len(indices)
        assignment_b2a = np.zeros((num_bonds, num_atoms), dtype=np.int64) # Maybe need too much memory
        assignment_a2b = np.zeros((num_atoms, num_bonds), dtype=np.int64) # Maybe need too much memory
        for i, idx in enumerate(indices):
            assignment_b2a[i, idx[1]] = 1
            assignment_a2b[idx[0], i] = 1           

        b2a_graph = coo_matrix(assignment_b2a)
        b2a_edges = list(zip(atom_graph.row, atom_graph.col))

        b2a_edges = torch.tensor(b2a_edges, dtype = torch.long).t().contiguous()
        num_bonds = torch.tensor(num_bonds, dtype = torch.long)
        num_atoms = torch.tensor(num_atoms, dtype = torch.long)
        x = torch.randn(num_atoms,1)
        #b2a_graph = Data(edge_index = b2a_edges, num_nodes = num_atoms, y = num_bonds)
        b2a_graph = Data(edge_index = b2a_edges, x = x, src_num_nodes=num_bonds, dst_num_nodes=num_atoms)

        ############################
        # build bond to bond graph #
        ############################
        bond_graph_base = assignment_b2a @ assignment_a2b
        np.fill_diagonal(bond_graph_base, 0) # eliminate self connections
        bond_graph_base[range(num_bonds), [indices.index([x[1],x[0]]) for x in indices]] = 0 
        x, y = np.where(bond_graph_base > 0)
        num_edges = len(x)

        #calculate angle
        angle_feat = np.zeros_like(x, dtype=np.float32)
        for i in range(num_edges):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            a = dist_mat[body1[0], body1[1]]
            b = dist_mat[body2[0], body2[1]]
            c = dist_mat[body1[0], body2[1]]
            if a == 0 or b == 0:
                print(body1, body2)
                print('One distance is zero.')
                angle_feat[i] = 0.
                return None, None
                # exit(-1)
            else:
                angle_feat[i] = cos_formula(a, b, c)
        
        #angle domain divisions
        unit = 180.0 / self.num_angle
        angle_index = (np.rad2deg(angle_feat) / unit).astype('int64')
        angle_index = np.clip(angle_index, 0, self.num_angle - 1)

       # multiple bond-to-bond graphs based on angle domains
        b2b_edges_list = [[] for _ in range(self.num_angle)]
        b2b_angle_list = [[] for _ in range(self.num_angle)]
        for i, (ind, radian) in enumerate(zip(angle_index, angle_feat)):
            b2b_edges_list[ind].append((x[i], y[i]))
            b2b_angle_list[ind].append(radian)
        
       # b2b_graph_list = [[] for _ in range(self.num_angle)]
        b2b_graph_list = []
        for ind in range(self.num_angle):
            b2b_edges_list[ind] = torch.tensor(b2b_edges_list[ind]).t().contiguous()
            b2b_angle_list[ind] = torch.tensor(b2b_angle_list[ind])            
            b2b_graph = Data(edge_index = b2b_edges_list[ind], num_nodes =num_bonds, edge_attr=b2b_angle_list[ind])
            b2b_graph_list.append(b2b_graph)

        #########################################
        # build index for inter-molecular bonds #
        #########################################
        bond_types = bond_pair_atom_types
        type_count = [0 for _ in range(len(pair_ids))]
        for type_i in bond_types:
            if type_i != -1:
                type_count[type_i] += 1

        bond_types = np.array(bond_types)
        type_count = np.array(type_count)
        a2a_graph.feats = torch.tensor(inter_feats,dtype=torch.long)
        a2a_graph.bond_types = torch.tensor(bond_types,dtype=torch.long)
        a2a_graph.type_count = torch.tensor(type_count,dtype=torch.long)
        
        graphs = a2a_graph, b2a_graph, b2b_graph_list
        
        return graphs
        
       
        
    def download(self):
        pass

    def process(self):
        """ Generate complex interaction graphs. """
       
        print('Processing raw protein-ligand complex data...')
        # file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
        with open(self.raw_paths[0], 'rb') as f:
            data_mols, data_Y = pickle.load(f)

        idx = 0
        #graphs = []
        for mol, y in tqdm(zip(data_mols, data_Y)):
            graphs = self._build_graph(mol)
            if graphs is None:
                continue
            #graphs.append(graph)

            
            y = torch.tensor(y,dtype=torch.float)
            graphs[0].y = y
            #print(graphs[1])
            
            data = graphs[0], graphs[1], graphs[2]
            # data2 = 
            # data3 = 
            #data = [graphs[0], graphs[1], graphs[2]]
        #self.labels = np.array(self.labels).reshape(-1, 1)
        # self.labels = np.array(data_Y).reshape(-1, 1)
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                    f'data_test_{idx}.pt'))
            elif self.val:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                    f'data_val_{idx}.pt'))

                                                        
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                    f'data_train_{idx}.pt'))
            idx += 1                        
                 
            
    def len(self):        
        
        return len(self.data[1])

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset            
                   """      
        
        if self.test:            
            dataa = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        elif self.val:            
            dataa = torch.load(os.path.join(self.processed_dir, 
                                 f'data_val_{idx}.pt'))

        else:
            dataa = torch.load(os.path.join(self.processed_dir, 
                                 f'data_train_{idx}.pt'))          
        return dataa
    


if __name__ == "__main__":
    from torch_geometric.nn import DataParallel
    from torch_geometric.data import Batch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import MessagePassing
    import os
    from torch_geometric.loader import DataLoader

    
    complex_data = ComplexDataset("/data/wujq/50",'/data/wujq', "g2016_train", 5, 6, 52,test = True)
    loader = DataListLoader(complex_data,
                        batch_size=1,
                        shuffle=False,
                        num_workers=4,
                        )
    for u in loader:
        for i in range(len(u[0][2])):
            if u[0][2][i].edge_index.shape[0] != 2: 
                print(1)
    
    