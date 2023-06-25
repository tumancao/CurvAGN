# Adapted from https://github.com/PaddlePaddle/PaddleHelix/blob/dev/apps/drug_target_interaction/sign/train.py
"""
Training process code for Curvature-based Adaptive Graph Neural Networks (CurvAGN).
"""
import os
import time
#import math
import argparse
import random
import numpy as np
from pcutils import rmse, mae, sd, pearson
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataListLoader
from parallel import DataParallel
from pcdataset import ComplexDataset
from pcmodel import SIGN

from tqdm import tqdm
from torch_geometric.loader import DataListLoader,DataLoader


#torch.seed(123)

def _set_up_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(torch.rand(1,3))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_hat_list = []
    y_list = []
    for batch_data in loader:
        a2a_g = batch_data[0].to(device)
        b2a_g = batch_data[1].to(device)
        b2b_gl = batch_data[2]
        b2b_gl = [data.to(device) for data in b2b_gl]
        y = a2a_g.y
        _, y_hat = model(a2a_g, b2a_g, b2b_gl)  
        y_hat_list += y_hat.tolist()
        y_list += y.tolist()

    y_hat = np.array(y_hat_list).reshape(-1,)
    y = np.array(y_list).reshape(-1,)
    
    return rmse(y, y_hat), mae(y, y_hat), sd(y, y_hat), pearson(y, y_hat)


def train(args, model, trn_loader, tst_loader, val_loader):
    epoch_step = len(trn_loader)   
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.dec_step, gamma = args.lr_dec_rate)
    
    model = model.to(device)
    rmse_val_best, res_tst_best = 1e9, ''
    running_log = ''
    print('Start training model...')
    for epoch in range(1, args.epochs + 1):
        sum_loss, sum_loss_inter = 0, 0
        model.train()
        start = time.time()
        for batch_data in tqdm(trn_loader):            
            a2a_g = batch_data[0].to(device)
            b2a_g = batch_data[1].to(device)
            b2b_gl = batch_data[2]
            b2b_gl = [data.to(device) for data in b2b_gl]
            y = (a2a_g.y).unsqueeze(dim=1)
            feats = a2a_g.feats 
            feats = feats.to(device)
            y = y.to(device)
            feats_hat, y_hat = model(a2a_g, b2a_g,b2b_gl)
            # loss function
            loss = F.l1_loss(y_hat, y, reduction='sum')
            loss_inter = F.l1_loss(feats_hat, feats, reduction='sum')
            loss += args.lambda_ * loss_inter
            loss.backward()
            optim.step()
            optim.zero_grad()
           # optim.clear_grad()
            scheduler.step()
    
            sum_loss += loss
            sum_loss_inter += loss_inter

        end_trn = time.time()
        
        rmse_val, mae_val, sd_val, r_val = evaluate(model, val_loader)
        rmse_tst, mae_tst, sd_tst, r_tst = evaluate(model, tst_loader)
        end_val = time.time()
        log = '-----------------------------------------------------------------------\n'
        log += 'Epoch: %d, loss: %.4f, loss_b: %.4f, time: %.4f, val_time: %.4f.\n' % (
                epoch, sum_loss/(epoch_step*args.batch_size), sum_loss_inter/(epoch_step*args.batch_size), end_trn-start, end_val-end_trn)
        log += 'Val - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_val, mae_val, sd_val, r_val)
        log += 'Test - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst)
        print(log)

        if rmse_val < rmse_val_best:
            rmse_val_best = rmse_val
            res_tst_best = 'Best - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n' % (rmse_tst, mae_tst, sd_tst, r_tst)
            if args.save_model:
                obj = {'model': model.state_dict(), 'epoch': epoch}
                path = os.path.join(args.model_dir, 'saved_model')
                torch.save(obj, path)
                # model.save(os.path.join(args.model_dir, 'saved_model'))

        running_log += log
        f = open(os.path.join(args.model_dir, 'log0423_3.txt'), 'w')
        f.write(running_log)
        f.close()

    f = open(os.path.join(args.model_dir, 'log0423_3.txt'), 'w')
    f.write(running_log + res_tst_best)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/data/wujq/')
    parser.add_argument('--dataset', type=str, default='g2016')
    parser.add_argument('--model_dir', type=str, default='/data/wujq/output/sign2/')
    parser.add_argument('--cuda', type=str, default='2')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--save_model", action="store_true", default=True)

    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=float, default= 1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--dec_step", type=int, default=18000)
    parser.add_argument('--stop_epoch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument("--num_convs", type=int, default= 2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--infeat_dim", type=int, default=36)
    parser.add_argument("--dense_dims", type=str, default='128*4,128*2,128')

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--cut_dist', type=float, default=5.)
    parser.add_argument('--num_angle', type=int, default=6)
    parser.add_argument('--merge_b2b', type=str, default='cat')
    parser.add_argument('--merge_b2a', type=str, default='mean')
    parser.add_argument('--num_flt', type=int, default=25)
    parser.add_argument('--root', type=str, default='/data/wujq/25/')

    args = parser.parse_args()
    args.activation = F.relu
    args.dense_dims = [eval(dim) for dim in args.dense_dims.split(',')]
    if args.seed:
        _set_up_seed(args.seed)

    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    
    if int(args.cuda) == -1:
        device = torch.device('cpu')
    else:
        device =torch.device('cuda:%s' % args.cuda)
    trn_complex = ComplexDataset(args.root, args.data_dir, "%s_train" % args.dataset, args.cut_dist, args.num_angle,args.num_flt)
    tst_complex = ComplexDataset(args.root, args.data_dir, "%s_test" % args.dataset, args.cut_dist, args.num_angle,args.num_flt,test=True)
    val_complex = ComplexDataset(args.root, args.data_dir, "%s_val" % args.dataset, args.cut_dist, args.num_angle,args.num_flt, val = True)
    trn_loader = DataLoader(trn_complex, args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    tst_loader = DataLoader(tst_complex, args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_complex, args.batch_size, shuffle=False, num_workers=0)
    

    model = SIGN(args)
    train(args, model, trn_loader, tst_loader, val_loader)