"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import numpy as np
from skimage.transform import rescale
import h5py
from skimage import color
from skimage import io
import glob
import csv
import os
from tqdm import tqdm

def create_point_cloud_depth(depth,fx,fy,cx,cy):
    depth_shape = depth.shape
    [x_d, y_d] = np.meshgrid(range(0,depth_shape[1]), range(0,depth_shape[0]))
    x3 = np.divide(np.multiply((x_d-cx),depth),fx)
    y3 = np.divide(np.multiply((y_d-cy),depth),fy)
    z3 = depth

    return np.stack((x3,y3,z3),axis=2)

def matrix3d2vector(matrix):
    return np.reshape(matrix,(-1,3))
def matrix2d2vector(matrix):
    return np.reshape(matrix,(-1))

def crop(img,new_size):
    img_size = np.shape(img)[0:2]
    start = (np.array(img_size) - np.array(new_size))/2
    end = (img_size-np.floor(start)).astype(np.int)
    start = np.ceil(start).astype(np.int)

    if len(np.shape(img)) == 3:
        return img[start[0]:end[0],start[1]:end[1],:]
    else:
        return img[start[0]:end[0],start[1]:end[1]] 
 
def read_list(path):

    with open(path) as f:
        lines = f.readlines()

    return [x.strip() for x in lines] 


def save_h5_scene_pc(filename, points, label):
    h5 = h5py.File(filename,"w")
    h5.create_dataset(
        "points", data = points,
        compression='gzip', compression_opts=4,
        dtype='float')
    h5.create_dataset(
        "label", data = label,
        dtype='uint8')


def write_csv(path,dictio):

    sw = csv.writer(open(path, "w"))
    for key, val in dictio.items():
        sw.writerow([key, val])

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":

    dataset_path = './nyu_v1'
    dataset_mat_path = dataset_path + '/nyu_depth_data_labeled.mat'
    

    path_nyuv1_h5  = dataset_path + "/h5_geometric/"
    create_folder(path_nyuv1_h5)


    
    train_split = read_list(dataset_path+"/list/train_list.txt")
    val_split = read_list(dataset_path+"/list/test_list.txt")
    scenes_types = read_list(dataset_path+"/list/scenes_labels.txt")
    dataset = train_split + val_split
    
    # intrinsics
    fx = 5.1930334103339817e+02;
    fy = 5.1816401430246583e+02;
    cx = 3.2850951551345941e+02;
    cy = 2.5282555217253503e+02;
    
    print("Loading .mat")
    f = h5py.File(dataset_mat_path, 'r')

    depths = np.transpose(np.asarray(f['depths']))
    scenes = np.transpose(np.asarray(f['scenes'])).squeeze()

    newsize =(400,560)
    
    for i in tqdm(dataset, ncols=100):
        filename = i
        i = int(i)-1
        
        depth = crop(depths[:,:,i],newsize)
    

        points = create_point_cloud_depth(depth,fx,fy,cx,cy)
        points_d8 = rescale(points, 1/8,anti_aliasing=False)
        
        label = (''.join(chr(j) for j in f[scenes[i]])).split("_0")[0]
        label_id = scenes_types.index(label)

        points_d8_vector = matrix3d2vector(points_d8)
        
        h5_name = filename+".h5" 
        
        save_h5_scene_pc(path_nyuv1_h5 + h5_name, points_d8_vector, label_id)
