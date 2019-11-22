RAGC: Residual Attention Graph Convolutional Network for Geometric 3D Scene Classification
=========

See our project website [here](https://imatge-upc.github.io/ragc/).

## Code Structure

* `./checkpoints/*` - Contains the weights used in the paper.
* `./dataset/` - Contains the scripts to generate the datasets. Moreover, this folder is used to save the dataset.
* `./torch_geometric_extension` - Contains some piece of code that extends the functionality of torch_geometric 
* `./results` - Folder used to save the results of train.py and test.py
* `./train_scripts` - Contains examples to train the different architectures, as it is done in the Article.
* `./test_scripts` - Contains examples to run the test.py using the provided checkpoints.
* `./*` - Contains the model, train and test scripts used to reproduce the results showed in the Article.

## Requirements
Install [Pytorch 1.2](https://pytorch.org) and [Pytorch-Geometric 1.3.1](https://github.com/rusty1s/pytorch_geometric) taking into account your cuda requirements. This code was tested using pyton 3.6 and cuda 10.

Run the following line ` pip3 install -r ./requirements.txt` to install the rest of the requirements.

## Datasets
### NYU_V1
Download the dataset from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v1.html). The data(a `.mat` file) needs to be located in `./dataset/nyu_v1`. Then execute the `dataset_preparation_nyu.py` script in order to prepare the data to be used.

### SUNRGBD
Download the dataset from [here](http://rgbd.cs.princeton.edu/). Uncompress the data and put it inside of `./dataset/sunrgbd`. Inside the folder you need to have a folder for each sensor of the dataset. Then execute the `dataset_preparation_sunrgbd.py` script in order to prepare the data to be used.

## Training
In order to train the network, the script `train.py` needs to be used. Please check the script to know which parameters are needed(at the beggining of the main function all the parameters are listed). Moreover, inside the folder `train_scripts` you can find some examples to use the `train.py` script. Take into account that some models have been splitted in different gpus. If you have a good gpu you can turn off this functionality and try to train the network in one gpu.

NOTE: Please check `graph_model.py` in order to understand how you can define your own architecture using the provided operations.
## Test
In order to run the test split using the provided checkpoints you need to use the `test.py` script.  Moreover, inside the folder `test_scripts` you can find some examples to use the `test.py` script.

## Issues
Due to the existence of some non-deterministic operations in pytorch as explained [here](https://pytorch.org/docs/stable/notes/randomness.html), some results may not be reproducible or give slightly different values. This effect is also reflected when you use different model of gpus to train and test the network.

## Citation
```
@InProceedings{Mosella-Montoro_2019_ICCV,
author = {Mosella-Montoro, Albert and Ruiz-Hidalgo, Javier},
title = {Residual Attention Graph Convolutional Network for Geometric 3D Scene Classification},
booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2019}
}
```

## Contact

For questions and suggestions send an e-mail to albert.mosella@upc.edu.
