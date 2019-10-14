"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
import torch_geometric
from h53dclass_dataloader import H53DClassDataset
from torch_geometric.data import Batch
import metrics

import graph_model as models
import argparse
from tqdm import tqdm
import ast
import os
import utils

def test(model, loader, loss_criterion, label_names, cuda = True):
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
    parser = argparse.ArgumentParser(description='agc')
    
    parser.add_argument('--model_config', default='', help='Defines the model as a sequence of layers')  
    parser.add_argument('--checkpoint_path_file', default='', help='Path of checkpoint to be used on the test')  

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--batch_parts', default=8, type=int, help='Batch parts.')

    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--multigpu', default=0, type=int, help='Bool, use multigpu')
    parser.add_argument('--lastgpu', default=0,type=int, help='Parameter to indicate what is the last gpu used')
    parser.add_argument('--nworkers', default=4, type=int, help='Num subprocesses to use for data loading. 0 means that the data will be loaded in the main process')
    
    # Dataset 
    parser.add_argument('--dataset_path', default ='', type=str, help = "Folder name that contains the h5 files")
    parser.add_argument('--dataset_folder', default = '', type=str, help = "Folder name that contains the h5 files")
    parser.add_argument('--test_split', default = 'list/test_list.txt', type=str, help = "Folder name that contains the h5 files")
    parser.add_argument('--nfeatures', default='1', help='Number of features of point clouds')
    
    # Results
    parser.add_argument('--exp_name', default='./results/cm_ragc_nyu.npy', help='Name of the experiment')
   

    # Filter generating network
    parser.add_argument('--fnet_widths', default='[16, 32]', help='List of width of hidden filter gen net layers (excluding the input and output ones, they are automatic)')
    parser.add_argument('--fnet_llbias', default=0, type=int, help='Bool, use bias in the last layer in filter gen net')
    parser.add_argument('--fnet_orthoinit', default=1, type=int, help='Bool, use orthogonal weight initialization for filter gen net.')
    parser.add_argument('--fnet_dropout', default=0, type=int, help='Int, use of dropout  for filter gen net.')
    parser.add_argument('--fnet_batchnorm', default=0, type=int, help='Bool, use of batchnorm for filter gen net.')
    parser.add_argument('--pc_attribs', default='', type=str, help='Edge attribute definition.')
    parser.add_argument('--coordnode', default=0, type=int, help='Put coordinates in node feature')
    # agc

    parser.add_argument('--agc_bias', default=0, type=int, help='Bool, use bias for edge conditioned convolution')
    
    args = parser.parse_args()
    args.fnet_widths = ast.literal_eval(args.fnet_widths)
    features = int(args.nfeatures)
    
    model = models.GraphNetwork(args.model_config, features, args.multigpu, 
                                args.fnet_widths, args.fnet_orthoinit,
                                args.fnet_llbias, default_edge_attr=args.pc_attribs,
                                default_agc_bias=args.agc_bias,
                                default_fnet_dropout=args.fnet_dropout,
                                default_fnet_batchnorm=args.fnet_batchnorm)
    loss_criterion = torch.nn.CrossEntropyLoss()
    if not args.multigpu:
        model.cuda(0)
    #creating DATABASE
    
    test_dataset = H53DClassDataset(args.dataset_path, args.dataset_folder, 
                                   args.test_split, coordnode=args.coordnode)


    test_loader = torch_geometric.data.DataLoader(test_dataset,
                                                batch_size = int(args.batch_size/args.batch_parts),
                                                num_workers = args.nworkers,
                                                shuffle = False,
                                                pin_memory=True
                                                )

    if (os.path.isfile(args.checkpoint_path_file)):
        print(model)
        checkpoint = torch.load(args.checkpoint_path_file)
        model.load_state_dict(checkpoint['model'])
    else:
        print('Checkpoint does not exists')
        exit
         

    label_path = os.path.join(args.dataset_path, 'list/scenes_labels.txt')
    if not os.path.isfile(label_path):
        raise RuntimeError("label file does not exists")
    label_names = utils.read_string_list(label_path)
    test_loss, test_cm = test(model, test_loader, 
                                loss_criterion, 
                                label_names, 
                                cuda = args.cuda)

    print('-> Test:\tAccuracy: {}, \tLoss: {}'.format(test_cm.accuracy(), test_loss, ))

    test_cm.save_npy(args.exp_name)
