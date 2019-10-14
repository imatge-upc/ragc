"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
import torch_geometric

import torch.nn as nn
import random
import numpy as np
import os
import sys
import math
import argparse
from tqdm import tqdm

import functools
import ast

import utils
import metrics
import graph_model as models
from tensorboardX import SummaryWriter
from h53dclass_dataloader import H53DClassDataset
import gc


def train(model, loader, loss_criteron, optimizer, label_names, batch_parts=0, cuda = True):
    model.train()
    numIt = len(loader)
    loader = tqdm(loader, ncols=100)
    acc = metrics.AverageMeter()
    losses = metrics.AverageMeter()
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Oranges')
    prev = -1
    optimizer.zero_grad()
    
    for i, batch in enumerate(loader, start = 0):
        if cuda: batch = batch.to('cuda:0')
        
        outputs = model(batch)

        out_device = outputs.device
        gt = batch.y.to(out_device)
        loss = loss_criterion(outputs, gt)
        loss.backward()
        
        batch_size = len(torch.unique(batch.batch))

        losses.update(loss.item(), batch_size)
        cm.add(gt.cpu().data.numpy(), outputs.cpu().data.numpy())
        
        if (i+1) % batch_parts == 0 or (i+1) == numIt:
            if batch_parts>1: 
                accum = i-prev
                prev = i
                for p in model.parameters():
                    p.grad.div_(accum)
            optimizer.step()
            optimizer.zero_grad()
                        
        torch.cuda.empty_cache()
    return losses.avg, cm

def val(model, loader, loss_criterion, label_names, cuda = True):
    model.eval()
    loader = tqdm(loader, ncols=100)
    losses = metrics.AverageMeter()
    cm = metrics.ConfusionMatrixMeter(label_names, cmap='Blues')
    
    with torch.no_grad():
        for i, batch in enumerate(loader, start = 0):
        
            if cuda: batch = batch.to('cuda:0')
            
            batch_size = len(batch.batch.unique())
                        
            outputs = model(batch)
            out_device = outputs.device
            gt = batch.y.to(out_device)

            loss = loss_criterion(outputs, gt)
            
            batch_size = len(torch.unique(batch.batch))

            losses.update(loss.item(), batch_size)

            cm.add(gt.cpu().data.numpy(), outputs.cpu().data.numpy())

            loader.set_postfix({"loss" : loss.item()})
            torch.cuda.empty_cache()

    return losses.avg, cm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='AGC')  
    
    # Optimization arguments
    parser.add_argument('--optim', default='adam', type=str, choices=['sgd','adam'], help='optimizer') 
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--betas', default='(0.9,0.999)', help = "Betas of adam optimizer")
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train. ')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--batch_parts', default=8, type=int, help='Batch parts')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--multigpu', default=0, type=int, help='Bool, use multigpu')
    parser.add_argument('--lastgpu', default=0,type=int, help='Parameter to indicate what is the last gpu used')
    parser.add_argument('--nworkers', default=4, type=int, help='Num subprocesses to use for data loading')
    
    # Dataset 
    parser.add_argument('--dataset_path', default ='', type=str, help = "Folder name that contains the h5 files")
    parser.add_argument('--dataset_folder', default = '', type=str, help = "Folder name that contains the h5 files")
    parser.add_argument('--train_split', default = 'list/train_list.txt', type=str, help = "Train split list path")
    parser.add_argument('--test_split', default = '/list/test_list.txt', type=str, help = "Test split list path")
    parser.add_argument('--nfeatures', default='1', help='Number of features of point clouds')
    parser.add_argument('--className',default='list/scenes_labels.txt', help = 'Path to the file that contains the name of the classes')
    parser.add_argument('--dataset', default='nyu_v1', help='Dataset name')    
    
    # Results
    parser.add_argument('--odir', default='./results', help='Directory to store results')
    parser.add_argument('--exp_name', default='Scene_Categorization', help='Name of the experiment')
   
    # Model    
    parser.add_argument('--model_config', default='', help='Defines the model as a sequence of layers.')  
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')        
    # Point cloud processing
    parser.add_argument('--pc_augm_input_dropout', default=0, type=float, help='Training augmentation: Probability of removing points in input point clouds')
    parser.add_argument('--pc_augm_scale', default=0, type=float, help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', default=1, type=int, help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', default=0.5, type=float, help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_attribs', default='', type=str, help='Edge attribute definition')
    parser.add_argument('--coordnode', default=0, type=int, help='Put coordinates in node feature')
    
    # Filter generating network
    parser.add_argument('--fnet_widths', default='[]', help='List of width of hidden filter gen net layers')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net')
    parser.add_argument('--fnet_dropout', default=0, type=int, help='Int, use of dropout  for filter gen net')
    parser.add_argument('--fnet_batchnorm', default=0, type=int, help='Bool, use of batchnorm for filter gen net')
    
    # agc

    parser.add_argument('--agc_bias', default=0, type=int, help='Bool, use bias in agc ')
    
    args = parser.parse_args()
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    args.betas = ast.literal_eval(args.betas)
    # Seeding
    utils.seed(args.seed)
    # creating experiment folder and init path for checkpoints, logs, etc
    exp_path = os.path.join(args.odir, args.dataset, args.dataset_folder, args.exp_name.replace(" ","_"))
    print("The experiment will save to " + exp_path)
    utils.create_folder(args.odir)
    utils.create_folder(exp_path)
    log_path = os.path.join(exp_path, 'log')
    utils.create_folder(log_path)
    log_train_path = os.path.join(log_path, 'train')
    utils.create_folder(log_train_path)
    log_test_path = os.path.join(log_path, 'test')
    utils.create_folder(log_test_path)
    checkpoint_path = os.path.join(exp_path, 'checkpoints')
    utils.create_folder(checkpoint_path)
    cm_path = os.path.join(exp_path, 'cm')
    utils.create_folder(cm_path)
    
    
    # saving command line used on this experiment
    with open(os.path.join(exp_path, 'cmdline.txt'), 'w') as f:
        f.write(" ".join(sys.argv))

    # Tensorboard writters
    
    
    train_writer = SummaryWriter(log_train_path)
    test_writer = SummaryWriter(log_test_path)

    # creating_model
    
    features = int(args.nfeatures)
    

    model = models.GraphNetwork(args.model_config, features, args.multigpu, 
                                args.fnet_widths, args.fnet_orthoinit,
                                args.fnet_llbias, default_edge_attr=args.pc_attribs,
                                default_agc_bias=args.agc_bias,
                                default_fnet_dropout=args.fnet_dropout,
                                default_fnet_batchnorm=args.fnet_batchnorm)


    if args.multigpu != 1 and args.cuda != 0: 
        model.to('cuda:0')
        print("GPU: ", torch.cuda.get_device_name(0))
    print(model)
    # optims 
    
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = args.momentum, weight_decay=args.wd)

    loss_criterion = torch.nn.CrossEntropyLoss()

    #creating DATABASE
    label_path = os.path.join(args.dataset_path, args.className)
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exists")
    label_names = utils.read_string_list(label_path)
    assert args.batch_size % args.batch_parts == 0    
    
    transform3d = {"dropout" : args.pc_augm_input_dropout, 
                    "scale" : args.pc_augm_scale,
                    "rot" : args.pc_augm_rot,
                    "mirror" : args.pc_augm_mirror_prob}

    train_dataset = H53DClassDataset(args.dataset_path, args.dataset_folder,
                                    args.train_split, transform3d = transform3d,
                                    coordnode = args.coordnode)

    test_dataset = H53DClassDataset(args.dataset_path, args.dataset_folder, 
                                   args.test_split,coordnode=args.coordnode)


    train_loader = torch_geometric.data.DataLoader(train_dataset,
                                                    batch_size = int(args.batch_size/args.batch_parts),
                                                    num_workers = args.nworkers,
                                                    shuffle = True,
                                                    drop_last = True,
                                                    pin_memory=True
                                                    )
    test_loader = torch_geometric.data.DataLoader(test_dataset,
                                                batch_size = int(args.batch_size/args.batch_parts),
                                                num_workers = args.nworkers,
                                                shuffle = False,
                                                pin_memory=True
                                                )
    


    is_best = False
    best_acc = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        
        print('Epoch {}/{} ({}):'.format(epoch, args.epochs, args.exp_name))

        train_loss, train_cm = train(model, train_loader,
                                            loss_criterion, 
                                            optimizer,
                                            label_names,
                                            batch_parts=args.batch_parts, 
                                            cuda = args.cuda)
        
        train_acc = train_cm.accuracy()

        gpu_mem_train = utils.max_gpu_allocated()
        train_writer.add_scalar('Loss', train_loss, epoch)
        train_writer.add_scalar('Accuracy', train_acc, epoch)
        train_writer.add_scalar('Learning_Rate', utils.get_learning_rate(optimizer)[0], epoch)
        
        for i in range(0, torch.cuda.device_count()):
            train_writer.add_scalar('GPU%02d_Memory' %i, gpu_mem_train[i], epoch)
        torch.cuda.empty_cache()
        print('-> Train:\tAccuracy: {}, \tLoss: {}'.format(train_acc, train_loss))

        
        test_loss, test_cm = val(model, test_loader, 
                                    loss_criterion, 
                                    label_names, 
                                    cuda = args.cuda)
        test_acc = test_cm.accuracy()

        gpu_mem_test = utils.max_gpu_allocated()
        test_writer.add_scalar('Loss', test_loss, epoch)
        test_writer.add_scalar('Accuracy', test_acc, epoch)

        for i in range(0, torch.cuda.device_count()):
            test_writer.add_scalar('GPU%02d_Memory' %i, gpu_mem_test[i], epoch)

        torch.cuda.empty_cache()
        print('-> Test:\tAccuracy: {}, \tLoss: {}'.format(test_acc, test_loss))
        is_best = test_acc > best_acc
        if is_best: 
            best_acc = test_acc
            test_writer.add_text('Best Accuracy', str(np.round(best_acc.item(), 2)), epoch)
        utils.save_checkpoint(epoch, model, optimizer, test_acc, is_best, best_acc, 
                             checkpoint_path, save_all=False)
        cm_file_name = os.path.join(cm_path, "cm_epoch_%i.npy" % epoch)
        test_cm.save_npy(cm_file_name)
        
        if math.isnan(test_loss) or math.isnan(train_loss): break
        
        del test_loss, test_acc, test_cm, train_loss, train_acc, train_cm, gpu_mem_train, gpu_mem_test 
        gc.collect()
    train_writer.close()
    test_writer.close()

