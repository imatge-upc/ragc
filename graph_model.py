"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
import torch
import torch_geometric
import torch_geometric.nn as nn_geometric
import torch.nn.init as init
import torch_geometric.transforms as T
import torch_geometric_extension as ext
from utils import gpuid2device

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import scatter_
from torch_cluster import nearest


def create_fnet(widths, nfeat, nfeato, orthoinit, llbias, dropout=None, batchnorm=False):
    fnet_modules = []
    for k in range(len(widths)-1):
        fnet_modules.append(torch.nn.Linear(widths[k], widths[k+1]))
        if orthoinit: init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        if batchnorm: fnet_modules.append(torch.nn.BatchNorm1d(widths[k+1]))
        fnet_modules.append(torch.nn.ReLU(True))
        if dropout != None and dropout != 0: fnet_modules.append(torch.nn.Dropout(dropout, inplace=False))
    fnet_modules.append(torch.nn.Linear(widths[-1], nfeat*nfeato, bias=llbias))
    if orthoinit: init.orthogonal_(fnet_modules[-1].weight)
    return torch.nn.Sequential(*fnet_modules)

def create_agc(nfeat, nfeato, fnet_widths, fnet_orthoinit, fnet_llbias, bias=False, 
               fnet_dropout = None, fnet_batchnorm = False, flow='source_to_target'):
    fnet = create_fnet(fnet_widths, nfeat, nfeato, fnet_orthoinit, fnet_llbias,
                       dropout=fnet_dropout, batchnorm=fnet_batchnorm)
    return ext.AGCConv(nfeat, nfeato, fnet, aggr="mean", flow=flow, bias=bias)


def create_dec(nfeat, nfeato, k, aggr='max'):
    aggr = aggr.lower()
    nn = torch.nn.Sequential(torch.nn.Linear(nfeat*2, nfeato, bias=False),
                            torch.nn.BatchNorm1d(nfeato),
                            torch.nn.LeakyReLU(negative_slope=0.2))

    return nn_geometric.DynamicEdgeConv(nn, k, aggr=aggr)



def conv1x1(in_feat, out_feat, stride=1):
    return torch.nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False)


def projection_shortcut(nfeat, nfeato):
    
    proj =  torch.nn.Sequential(torch.nn.Conv1d(nfeat, nfeato, kernel_size=1),
                        torch.nn.BatchNorm1d(nfeato))

    return proj

class ResidualDecBlock(torch.nn.Module):

    def __init__(self, nfeat, nfeato, k, aggr='max'):
        super(ResidualDecBlock, self).__init__()
        
        self.conv1 = create_dec(nfeat, nfeato, k, aggr=aggr)
        self.bn1 = torch.nn.BatchNorm1d(nfeato, affine=True)
        self.relu = torch.nn.ReLU(True)
        self.conv2 = create_dec(nfeato, nfeato, k, aggr=aggr)
        self.bn2 = torch.nn.BatchNorm1d(nfeato, affine=True)
        
        if nfeat != nfeato: self.chan_adapt_criteria = projection_shortcut(nfeat, nfeato)
        else: self.chan_adapt_criteria=None

    def forward(self, x, batch):
        shape = x.shape
        identity = x
        x = self.conv1(x, batch)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, batch)
        x = self.bn2(x)

        if self.chan_adapt_criteria is not None:
            identity = identity.view(1, shape[1], shape[0], 1).squeeze(-1)
            identity = self.chan_adapt_criteria(identity)
            identity = identity.view(shape[0], -1)
        x += identity
        return self.relu(x)


class ResidualAGCBlock(torch.nn.Module):

    def __init__(self, nfeat, nfeato, fnet_widths, fnet_orthoinit, fnet_llbias, bias=False, fnet_dropout=None, fnet_batchnorm=False, flow='source_to_target'):
        super(ResidualAGCBlock, self).__init__()
        
        self.conv1 = create_agc(nfeat, nfeato, fnet_widths, fnet_orthoinit, fnet_llbias, 
                                bias=bias,
                                fnet_dropout=fnet_dropout,
                                fnet_batchnorm=fnet_batchnorm,
                                flow=flow)
        self.bn1 = torch.nn.BatchNorm1d(nfeato, affine=True)
        self.relu = torch.nn.ReLU(True)
        self.conv2 = create_agc(nfeato, nfeato, fnet_widths, fnet_orthoinit, fnet_llbias, 
                                bias=bias,
                                fnet_dropout=fnet_dropout,
                                fnet_batchnorm=fnet_batchnorm,
                                flow=flow)
        self.bn2 = torch.nn.BatchNorm1d(nfeato, affine=True)
        
        if nfeat != nfeato: self.chan_adapt_criteria = projection_shortcut(nfeat, nfeato)
        else: self.chan_adapt_criteria=None

    def forward(self, x, edge_index, edge_attr):
        shape = x.shape
        identity = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)

        if self.chan_adapt_criteria is not None:
            identity = identity.view(1, shape[1], shape[0], 1).squeeze(-1)
            identity = self.chan_adapt_criteria(identity)
            identity = identity.view(shape[0], -1)
        x += identity
        return self.relu(x)



class ResidualFeastBlock(torch.nn.Module):

    def __init__(self, nfeat, nfeato, M, bias=True, flow='source_to_target'):
        super(ResidualFeastBlock, self).__init__()
        
        self.conv1 = nn_geometric.FeaStConv(nfeat, nfeato, M, bias=bias, flow=flow)
        self.bn1 = torch.nn.BatchNorm1d(nfeato, affine=True)
        self.relu = torch.nn.ReLU(True)
        self.conv2 = nn_geometric.FeaStConv(nfeato, nfeato, M, bias=bias, flow=flow)
        self.bn2 = torch.nn.BatchNorm1d(nfeato, affine=True)
        
        if nfeat != nfeato: self.chan_adapt_criteria = projection_shortcut(nfeat, nfeato)
        else: self.chan_adapt_criteria=None

    def forward(self, x, edge_index):
        shape = x.shape
        identity = x
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)

        if self.chan_adapt_criteria is not None:
            identity = identity.view(1, shape[1], shape[0], 1).squeeze(-1)
            identity = self.chan_adapt_criteria(identity)
            identity = identity.view(shape[0], -1)
        x += identity
        return self.relu(x)


def numberEdgeAttr(edge_attr, nfeat):
    
    nEA = 0
    if edge_attr != None:
        if type(edge_attr) == str: 
            edge_attr = [edge_attr]
        for attr in edge_attr:
            attr = attr.strip().lower()
            if attr == 'poscart':
                nEA = nEA + 3
            
            elif attr == 'posspherical':
                nEA = nEA + 3

            else:
                raise RuntimeError('{} is not supported'.format(attr))

    return nEA

class GraphReg(torch.nn.Module):
    def __init__(self, n_neigh = 9, rad_neigh=0.1, knn=None, self_loop = True,
    edge_attr = None, flow='source_to_target'):
        super(GraphReg, self).__init__() 
        # defining graph transform
        graph_transform_list = []
        self.del_edge_attr = False
        self.knn = knn
        self.n_neigh = n_neigh
        self.rad_neigh = rad_neigh
        self.self_loop = self_loop
        self.edge_attr = edge_attr
        if self.knn == True:
            graph_transform_list.append(T.KNNGraph(n_neigh, loop = self_loop,
            flow=flow))

        elif self.knn == False:
            graph_transform_list.append(T.RadiusGraph(self.rad_neigh, loop = self_loop,
                                                      max_num_neighbors = n_neigh,
                                                      flow=flow))
        else:
            print("Connectivity of the graph will not be re-generated")  

        # edge attr
        if edge_attr is not None:
            self.del_edge_attr = True
            if type(edge_attr) == str:
                if edge_attr:
                    edge_attr = [attr.strip() for attr in edge_attr.split('-')]
                else:
                    edge_attr=[]
            for attr in edge_attr:
                attr = attr.strip().lower()
                
                if attr == 'poscart':
                    graph_transform_list.append(T.Cartesian(norm=False, cat=True))
                
                elif attr == 'posspherical':
                    graph_transform_list.append(ext.Spherical(norm=False, cat=True))

                else:
                    raise RuntimeError('{} is not supported'.format(attr))
        self.graph_transform = T.Compose(graph_transform_list)

    def forward(self, data):
        if self.del_edge_attr: data.edge_attr = None
        data = self.graph_transform(data)
        return data

    def extra_repr(self):
            s = "knn={knn}"
            if self.knn == True:
              s += ", n_neigh={n_neigh}"  
              s += ", self_loop={self_loop}"
            elif self.knn == False:
              s += ", n_neigh={n_neigh}"  
              s += ", rad_neigh={rad_neigh}"  
              s += ", self_loop={self_loop}"

            s += ", edge_attr={edge_attr}"
            return s.format(**self.__dict__)


class GraphPooling(torch.nn.Module):
    def __init__(self, pool_rad, n_neigh=9, rad_neigh=0.1, knn=True,
                self_loop=True, aggr='max', edge_attr=None, flow='source_to_target'):
        super(GraphPooling, self).__init__() 
        self.pool_rad = pool_rad
        self.graph_reg = GraphReg(n_neigh, rad_neigh, knn, self_loop, edge_attr, flow=flow)
        
        self.aggr = aggr.strip().lower()
        
        if aggr == 'max':
            self._aggr = 'max'
        elif aggr == 'avg':
            self._aggr = 'mean'
        else:
            raise RuntimeError("Invalid aggregation method in Graph Pooling Layer")
        
    def forward(self, data):
        
        cluster = nn_geometric.voxel_grid(data.pos, data.batch, self.pool_rad, 
                                          start=data.pos.min(dim=0)[0] - self.pool_rad * 0.5, 
                                          end=data.pos.max(dim=0)[0] + self.pool_rad * 0.5)
        
        cluster, perm = consecutive_cluster(cluster)
 
        new_pos = scatter_('mean', data.pos, cluster)
        new_batch = data.batch[perm]
  
        cluster = nearest(data.pos, new_pos, data.batch, new_batch)
        data.x = scatter_(self._aggr, data.x, cluster, dim_size=new_pos.size(0))

  
        data.pos = new_pos
        data.batch = new_batch
        data.edge_attr = None
        
        
        data = self.graph_reg(data)
        return data

    def extra_repr(self):
        s = 'aggr={aggr}'
        s += ', pool_rad={pool_rad}'
        return s.format(**self.__dict__)


    
class GlobalGraphPooling(torch.nn.Module):
    def __init__(self, aggr):
        super(GlobalGraphPooling, self).__init__()
        self.aggr = aggr
        if self.aggr == 'max':
            self.pool = nn_geometric.global_max_pool
        elif self.aggr == 'avg':
            self.pool = nn_geometric.global_mean_pool

        else:
            raise RuntimeError("Invalid aggration method in Global Graph Pooling Layer")
    def forward(self, x, batch):
        return self.pool(x, batch)
    
    def extra_repr(self):
         s = 'aggr={aggr}'
         return s.format(**self.__dict__)

class GraphNetwork(torch.nn.Module):
    def __init__(self, config, nfeat, multigpu, fnet_widths, fnet_orthoinit,
                fnet_llbias, default_edge_attr = 'poscart', 
                default_agc_bias=False,
                default_gn_groups=1, default_gn_cpg=0,
                default_fnet_dropout=None, default_fnet_batchnorm=False):
        super(GraphNetwork, self).__init__() 
        self.multigpu = multigpu
        self.devices = []
        self.flow = 'source_to_target'
        nfeat = nfeat
        nEdgeAttr = 0
        

        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')
            device = None
            if default_edge_attr:
                edge_attr = [attr.strip() for attr in default_edge_attr.split('-')]
            else:
                edge_attr=[]
            fnet_dropout = default_fnet_dropout
            fnet_batchnorm = default_fnet_batchnorm
            agc_bias = default_agc_bias
            # Graph Generation
            if conf[0] == 'ggknn':
                if len(conf) < 2: raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                neigh = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit():
                        device = conf[2]
                    else:
                        edge_attr = [attr.strip() for attr in conf[2].split('-')]
                        if len(conf) == 4: device = conf[3]
                        elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ggknn layer".format(d))
                
                module = GraphReg(knn=True, n_neigh=neigh, edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'ggrad':
                if len(conf) < 3: raise RuntimeError("{} Graph Generation layer requires more arguments".format(d))
                rad = float(conf[1])
                neigh = int(conf[2])
                if len(conf) > 3:
                    if conf[3].isdigit():
                        device = conf[3]
                    else:
                        edge_attr = [attr.strip() for attr in conf[3].split('-')]
                        if len(conf) == 5: device = conf[4]
                        elif len(conf) > 5: raise RuntimeError("Invalid parameters in {} ggrad layer".format(d))
                
                module = GraphReg(knn=False, n_neigh=neigh, edge_attr=edge_attr, self_loop=True, flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)

            # Fully connected layer
            # Args: output_features
            elif conf[0] == 'f':
                if len(conf) < 2: raise RuntimeError("{} Fully connected layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                module = torch.nn.Linear(nfeat, nfeato)
                nfeat = nfeato
                if len(conf) == 3: device = conf[2]
                elif len(conf) > 3: raise RuntimeError("Invalid parameters in {} fully connected layer".format(d))

            # Batch norm layer
            elif conf[0] == 'b':
                module = torch.nn.BatchNorm1d(nfeat, affine=True, track_running_stats=True)
                if len(conf) == 2: device = conf[1]
                elif len(conf) > 3: raise RuntimeError("Invalid parameters in {} batchnom layer".format(d))
            
            # Relu layer
            elif conf[0] == 'r':
                module = torch.nn.ReLU(True)
                if len(conf) == 2: device = conf[1]
                elif len(conf) > 3: raise RuntimeError("Invalid parameters in {} relu layer".format(d))
            # Dropout
            elif conf[0] == 'd':
                if len(conf) < 2: raise RuntimeError("{} Dropout layer requires as argument the probabity to zeroed an element".format(d))
                prob = float(conf[1])
                module = torch.nn.Dropout(prob, inplace=False)
                if len(conf) == 3: device = conf[2]
                elif len(conf) > 3: raise RuntimeError("Invalid parameters in {} dropout layer".format(d))
           
            #Residual Dagconv
            elif conf[0] == 'rdec':
                if len(conf) < 3: raise RuntimeError("{} DEC layer requires as argument the output features".format(d))
                nfeato=int(conf[1])
                k=int(conf[2])
                if len(conf) == 4: device = conf[3]
                elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
                
                module=ResidualDecBlock(nfeat, nfeato, k, aggr='max')
                nfeat = nfeato
            
            #Dagconv
            elif conf[0] == 'dec':
                if len(conf) < 3: raise RuntimeError("{} DEC layer requires as argument the output features".format(d))
                nfeato=int(conf[1])
                k=int(conf[2])
                if len(conf) == 4: device = conf[3]
                elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
                
                module=create_dec(nfeat, nfeato, k, aggr='max')
                nfeat = nfeato
           
            #FeaStConv
            elif conf[0] == 'feast':
                if len(conf) < 3: raise RuntimeError("{} FeastConv layer requires as argument the output features".format(d))
                nfeato=int(conf[1])
                M=int(conf[2])
                if len(conf) == 4: device = conf[3]
                elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
                
                module=nn_geometric.FeaStConv(nfeat, nfeato, M, bias=True)
                nfeat = nfeato
           
            #Residual FeaStConv
            elif conf[0] == 'rfeast':
                if len(conf) < 3: raise RuntimeError("{} FeastConv layer requires as argument the output features".format(d))
                nfeato=int(conf[1])
                M=int(conf[2])
                if len(conf) == 4: device = conf[3]
                elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
                
                module=ResidualFeastBlock(nfeat, nfeato, M, bias=True)
                nfeat = nfeato
            
            # agc
            elif conf[0] == 'agc':
                if len(conf) < 2: raise RuntimeError("{} agc layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit(): device = conf[2]
                    else:
                        params = [param.strip() for param in conf[2].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                agc_bias = int(p[1])
                            elif param == 'fdropout':
                                fnet_dropout = int(p[1])
                            elif param == 'fbatchnorm':
                                fnet_batchnorm = int(p[1])


                        if len(conf) == 4: device = conf[3]
                    
                        elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
            
                module = create_agc(nfeat, nfeato, [nEdgeAttr] + fnet_widths,
                                    fnet_orthoinit, fnet_llbias,
                                    bias=agc_bias,
                                    fnet_dropout=fnet_dropout,
                                    fnet_batchnorm=fnet_batchnorm,
                                    flow=self.flow)
                nfeat = nfeato

            elif conf[0] == 'ragc':
                if len(conf) < 2: raise RuntimeError("{} agc layer requires as argument the output features".format(d))
                nfeato = int(conf[1])
                if len(conf) > 2:
                    if conf[2].isdigit(): device = conf[2]
                    else:
                        params = [param.strip() for param in conf[2].split('-')]
                        for param in params:
                            p = [p.strip() for p in param.split(':')]
                            param = p[0]
                            if param == 'bias':
                                agc_bias = int(p[1])
                            elif param == 'fdropout':
                                fnet_dropout = int(p[1])
                            elif param == 'fbatchnorm':
                                fnet_batchnorm = int(p[1])


                        if len(conf) == 4: device = conf[3]
                    
                        elif len(conf) > 4: raise RuntimeError("Invalid parameters in {} ec layer".format(d))
            
                module = ResidualAGCBlock(nfeat, nfeato, [nEdgeAttr] + fnet_widths,
                                    fnet_orthoinit, fnet_llbias,
                                    bias=agc_bias,
                                    fnet_dropout=fnet_dropout,
                                    fnet_batchnorm=fnet_batchnorm,
                                    flow=self.flow)
                nfeat = nfeato




         # KNN pooling layer agc
            elif conf[0] == 'pknn':
                if len(conf) < 4: raise RuntimeError("{} KNN Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                nn = int(conf[3])
                if len(conf) > 4:
                    if conf[4].isdigit():
                        device = conf[4]
                    else:
                        edge_attr = [attr.strip() for attr in conf[4].split('-')]
                        if len(conf) == 6: device = conf[5]
                        elif len(conf) > 6: raise RuntimeError("Invalid parameters in {} pknn layer".format(d))
                
                module = GraphPooling(pradius, n_neigh=nn,
                                                knn=True, self_loop=True, 
                                                aggr=aggr, edge_attr=edge_attr,
                                                flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)

            # Radius pooling layer  
            elif conf[0] == 'prnn':
                if len(conf)<5: raise RuntimeError("{} RNN Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])
                rad_neigh = float(conf[3])
                nn = int(conf[4])

                if len(conf) > 5:
                    if conf[5].isdigit():
                        device = conf[5]
                    else:
                        edge_attr = [attr.strip() for attr in conf[5].split('-')]
                        if len(conf) == 7: device = conf[6]
                        elif len(conf) > 7: raise RuntimeError("Invalid parameters in {} prnn layer".format(d))
               
                module = GraphPooling(pradius, n_neigh=nn, rad_neigh=rad_neigh, 
                                                knn=False, self_loop=True, aggr=aggr, 
                                                edge_attr=edge_attr,
                                                flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)
            
            elif conf[0] == 'p':
                if len(conf)<3: raise RuntimeError("{} RNN Pool layer requires more arguments".format(d))
                aggr = conf[1]
                pradius = float(conf[2])

                if len(conf) == 4 and conf[3].isdigit(): device = conf[3]
                else: raise RuntimeError("Invalid parameters in {} prnn layer".format(d))

                module = GraphPooling(pradius, n_neigh=None, rad_neigh=None, 
                                    knn=None, self_loop=True, aggr=aggr, 
                                    edge_attr=None,
                                    flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)

            elif conf[0] == 'gp':
                if len(conf) < 2: raise RuntimeError("Global Pooling Layer needs more arguments")
                aggr = conf[1]
                module = GlobalGraphPooling(aggr)
                
                if len(conf) == 3:
                    device = conf[2] 
                module = GlobalGraphPooling(aggr)
            # change edge atribs
            elif conf[0] == 'eg':
                if len(conf) > 1:
                    if conf[1].isdigit():
                        device = gconf[1]
                
                    else:
                        edge_attr = [attr.strip() for attr in conf[1].split('-')]
                        if len(conf) == 3: device = conf[2]
                        elif len(conf) > 3: raise RuntimeError("Invalid parameters in {} edge_generation layer".format(d))

                module = GraphReg(knn=None, edge_attr=edge_attr, flow=self.flow)
                nEdgeAttr = numberEdgeAttr(edge_attr, nfeat)

            else:
                raise RuntimeError("{} layer does not exist".format(conf[0]))
            # Adding layer to modules
            if self.multigpu == True:
                if device == None: raise RuntimeError("Multigpu is enabled and layer {} does not have gpu assigned.".format(d))
                device = gpuid2device(device)
                self.devices.append(device)
                module = module.to(device)
            self.add_module(str(d), module)

    def forward(self, data):
        for i, module in enumerate(self._modules.values()):
            if self.multigpu: data.to(self.devices[i])
            if type(module) == torch.nn.Linear or \
               type(module) == torch.nn.BatchNorm1d or \
               type(module) == torch.nn.Dropout or \
               type(module) == torch.nn.ReLU:
                if (type(data) == torch_geometric.data.batch.Batch or
                    type(data) == torch_geometric.data.data.Data):
                    
                    data.x = module(data.x)

                elif (type(data) == torch.Tensor):

                    data = module(data)

                else: 
                    raise RuntimeError("Unknonw data type in forward time in {} module".format(type(module)))

            
            elif type(module) == ext.AGCConv or \
                 type(module) == ResidualAGCBlock:
                data.x = module(data.x, data.edge_index, data.edge_attr.float())
            
            elif type(module) == ResidualFeastBlock or \
                 type(module) == nn_geometric.FeaStConv:
                data.x=module(data.x, data.edge_index)


            elif type(module) == GraphPooling or\
                 type(module) == GraphReg:
                
                data = module(data)

            elif type(module) == GlobalGraphPooling:

                data = module(data.x, data.batch)
            
            elif type(module) == ResidualDecBlock or\
                 type(module) == nn_geometric.DynamicEdgeConv:

                data.x = module(data.x, data.batch)

            else:
                raise RuntimeError("Unknown Module in forward time")
        
        return data
