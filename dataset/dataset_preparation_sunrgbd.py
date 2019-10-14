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

def read_depth_sunrgbd(depth_path):
    depthVis = io.imread(depth_path)
    depthInPaint = ((depthVis >> 3) | (depthVis << 13))/1000
    depthInPaint[np.where(depthInPaint>8)]=8
    return depthInPaint

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


def crop(img,new_size):
    img_size = np.shape(img)[0:2]
    
    start = (np.array(img_size) - np.array(new_size))/2
    end = (img_size-np.floor(start)).astype(np.int)
    start = np.ceil(start).astype(np.int)
    
    if len(np.shape(img)) == 3:
        return img[start[0]:end[0],start[1]:end[1],:] 
    else:
        return img[start[0]:end[0],start[1]:end[1]] 


if __name__ == "__main__":
    
    ROOT_PATH = './sunrgbd'
    
    images_list = ROOT_PATH + '/list/sun_list.txt'
    label_list = ROOT_PATH + '/list/scenes_labels.txt'

    path_sunrgbd_h5 = ROOT_PATH + '/h5_geometric/'
    create_folder(path_sunrgbd_h5)
    


    images = read_list(images_list)
    
    dataset_labels = read_list(label_list)
    newSize = (400,560)
    
    for i in tqdm(range(0,len(images)),ncols = 100):
        img_folder = ROOT_PATH+"/"+images[i]
        depth_path = glob.glob(img_folder+"/depth_bfx/*.png")[0]
        depth_img = read_depth_sunrgbd(depth_path)
        intrinsic = np.loadtxt(img_folder + '/intrinsics.txt')
        readlabel = open(img_folder + "/scene.txt","r")
        label = readlabel.read()
    
        points = create_point_cloud_depth(depth_img, intrinsic.item(0), intrinsic.item(4), intrinsic.item(2), intrinsic.item(5))
        
        points_cropped = crop(points,newSize)
        
        points_d8 = rescale(points_cropped, 1/8,anti_aliasing=False)
       
    
        # Save h5_file
        h5_name = (images[i].replace("/","_")+".h5").replace("_.h5",".h5") 

        label_id = dataset_labels.index(label)
        points_d8_vector = matrix3d2vector(points_d8)
        
        save_h5_scene_pc(path_sunrgbd_h5  + h5_name, points_d8_vector, label_id)
