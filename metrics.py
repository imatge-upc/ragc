"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import textwrap
import re, os, sys

class ConfusionMatrixMeter():
    def __init__(self, labels, cmap='orange'):
        self._cmap = cmap
        self._k = len(labels)
        self._labels = labels
        self._cm = np.ndarray((self._k, self._k), dtype=np.int32)
        self.reset()
    
    def reset(self):
        self._cm.fill(0)
    def add(self, target, predicted):
        assert predicted.shape[0] == target.shape[0], \
        'number of targets and predicted outputs do not match'
        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self._k, \
            'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self._k) and (predicted.min() >= 0), \
            'predicted values are not between 1 and k'
        self._cm += confusion_matrix(target,predicted,range(0,self._k))
    def value(self,normalize=False):
        if normalize:
            np.set_printoptions(precision=2)
            return np.divide(self._cm.astype('float'), self._cm.sum(axis=1).clip(min=1e-12)[:, np.newaxis])
        else:
            return self._cm
    def accuracy(self):
        return np.divide(self.value().trace(), self.value().sum())*100

    def F1_Score(self):
        """ Computes mean F1 score from confusion matrix, weighted by class support (for Sydney) """
        avgwf1 = 0
        N = self._cm.shape[0]
        f1 = np.zeros(N)
        for i in range(N):
            pr = self._cm[i][i] / max(1,np.sum(self._cm[:,i])) 
            re = self._cm[i][i] / max(1,np.sum(self._cm[i]))

            f1[i] = 2*pr*re/max(1,pr+re)

            avgwf1 = avgwf1 + np.sum(self._cm[i]) * f1[i]
        return avgwf1 / self._cm.sum()   

    def save_npy(self, filename, normalize=False):
        np.save(filename,self.value())

def accuracy(output, target, topk=(1,)):
    """ accuracy using pytorch functionalities
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() # transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.val = val
        self.sum += val*count
        self.count += count
        self.avg = self.sum / self.count



