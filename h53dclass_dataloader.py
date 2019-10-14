"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
from torch_geometric.data import Data
import math
import random
import numpy as np
import transforms3d
import os
import h5py


def read_string_list(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()
    return [x.strip() for x in lines]

def dropout(P,F,p):
    """ Removes points with probability p from vector of points and features"""
    idx = random.sample(range(P.shape[0]), int(math.ceil((1-p)*P.shape[0])))
    return P[idx,:], F[idx,:] if F is not None else None


class H53DClassDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, h5_folder, split,
                transform3d=None, coordnode=False):
        self.root_path = root_path

        self.h5_path = os.path.join(self.root_path, h5_folder)
        self.split = read_string_list(os.path.join(self.root_path, split))
        self.coordnode = coordnode
        self.transform3d = transform3d
    def __getitem__(self, index):
        h5_file = h5py.File(os.path.join(self.h5_path, self.split[index]+".h5"), 'r')
        cls = int(np.asarray((h5_file["label"])))
        P = np.asarray(h5_file["points"])
        F = np.ones((len(P),1), dtype=np.float32)
        if self.transform3d is not None:
            if self.transform3d["dropout"] > 0:
                P, F = dropout(P, F, self.transform3d["dropout"])
            M = np.eye(3)
            if self.transform3d["scale"] > 1:
                s = random.uniform(1/self.transform3d["scale"], self.transform3d["scale"])
                M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
            if self.transform3d["rot"]:
                angle = random.uniform(0, 2*math.pi)
                M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # z=upright assumption

            if self.transform3d["mirror"] > 0:
                if random.random() < self.transform3d["mirror"]/2: 
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
                if random.random() < self.transform3d["mirror"]/2: 
                    M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
            P = np.dot(P, M.T)

        P-=np.min(P, axis=0)

        if not self.coordnode:
            data = Data(x = torch.tensor(F), pos = torch.tensor(P), y = cls)
        else:
            data = Data(x = torch.tensor(P, dtype = torch.float32), pos = torch.tensor(P), y = cls)
        
        return data

    def __len__(self):
        return len(self.split)
