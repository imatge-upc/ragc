"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
import random
import numpy as np
import shutil
import os 


def gpuid2device(id):
    return 'cuda:{}'.format(id)

def read_string_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()
    return [x.strip() for x in lines]

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def max_gpu_allocated():
    mem = []
    for i in range(0, torch.cuda.device_count()):
        mem.append(np.divide(torch.cuda.max_memory_allocated(i), 1e9))
    return mem

def seed(seed):
    """ Seeding pytorch, numpy, random and imgaug frameworks
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # it isn't needed to check if cuda is available
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_rng_states():
    np_state = np.random.get_state()
    ran_state = random.getstate()
    torch_state = torch.get_rng_state()
    torch_cuda_state=[]
    for i in range(0, torch.cuda.device_count()):
        torch_cuda_state.append(torch.cuda.get_rng_state(i))
    return  {'numpy': np_state, 'random': ran_state, 
             'torch': torch_state, 'torch_cuda': torch_cuda_state}

def set_rng_states(states):
    np.random.set_state(states["numpy"])
    random.setstate(states["random"])
    torch.set_rng_state(states["torch_state"])
    for i, state in enumerate(states["torch_cuda"]):
        torch.cuda.set_rng_state(state, i)

def save_checkpoint(epoch, model, optimizer, acc, is_best, best_acc, 
                    output = './res/checkpoint', save_all=False):
    filename = os.path.join(output, 'checkpoint')
    rng_states = get_rng_states()
    devices = []
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                devices.append(state[k].device)

    checkpoint = { 'epoch' : epoch, 
              'model' : model.state_dict(), 
              'optimizer' : optimizer.state_dict(),
              'optimizer_devices' : devices,
              'acc' : best_acc,
              'best_acc' : best_acc,
              'rng_states' : rng_states}
    
    torch.save(checkpoint, filename + '_latest.pth.tar')
    if save_all:
        shutil.copyfile(filename + '_latest.pth.tar',
                        filename + '_epoch_%i.pth.tar' % epoch)
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', 
                        filename +'_best.pth.tar')
def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model_state = checkpoint['model']
    optimizer_state = checkpoint['optimizer']
    
    
    set_rgn(checkpoint['rng_states'])
    
    return epoch, best_acc, model_state, optimizer_state

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


