"""
	Residual Attention Graph Convolutional network for Geometric 3D Scene Classification
    2019 Albert Mosella-Montoro <albert.mosella@upc.edu>
"""
from math import pi as PI
import torch
class Spherical(object):

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        cart = pos[col] - pos[row]

        rho = torch.norm(cart, p=2, dim=-1)

        theta = torch.atan2(cart[..., 1], cart[..., 0]).view(-1, 1)
        theta = theta + (theta < 0).type_as(theta) * (2 * PI)

        phi = torch.acos(cart[..., 2] / (rho + 1e-6)).view(-1, 1)

        spher = torch.cat([rho.view(-1, 1), theta, phi], dim=-1)


        if self.norm:
            rho = rho / (rho.max() if self.max is None else self.max)
            theta = theta / (2 * PI)
            phi = phi / PI
        
        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, spher.type_as(pos)], dim=-1)
        else:
            data.edge_attr = spher
        
        return data
        """
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        assert pos.dim() == 2 and pos.size(1) == 3

        offsets = pos[col] - pos[row]
        
        p1 = torch.norm(offsets,p=2, dim=-1)
        p2 = torch.atan2(offsets[...,1], offsets[...,0]).view(-1,1)
        p3 = torch.acos(offsets[...,2] / (p1 + 1e-6)).view(-1,1)
        p1 = p1.view(-1,1)


        if self.norm:
            pass
            #rho = rho / (rho.max() if self.max is None else self.max)
            #theta = theta / (2 * PI)
            #phi = phi / PI

        polar = torch.cat([p1, p2, p3], dim=-1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, polar.type_as(pos)], dim=-1)
        else:
            data.edge_attr = polar
        return data
        """
    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)
