import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GNN(nn.Module):
    def __init__(self, n_F, n_gnn = 1):
        '''
            Time complexity: O(NNF+NFF)
        '''
        super().__init__()
        self.relu = nn.ReLU()
        self.n_gnn = n_gnn
        self.W = nn.ParameterList([ nn.Parameter(torch.empty((n_F, n_F))) for _ in range(n_gnn)])
        self.bn = nn.ModuleList([ LayerNorm(n_F) for _ in range(n_gnn) ])

        ## init self.W
        for i in range(n_gnn):
            torch.nn.init.xavier_uniform_(self.W[i])
    
    def forward(self, A, X, reverse=False):
        '''helper:
            Time complexity: O(NNF+NFF)
            A: adjacent matrix bs, N, N
            X: features bs, N, F

            return bs, N, F
        '''
        A = nn.Softmax(dim=-1)(A) ## Laplacian Smoothing http://www4.comp.polyu.edu.hk/~csxmwu/papers/AAAI-2018-GCN.pdf
        if reverse: A = 1.0 - A
        for i in range(self.n_gnn):
            node = torch.matmul(A, self.bn[i](X) )
            X = self.relu(torch.matmul(node, self.W[i])) 
        return X

class SpatialGNN(nn.Module):
    def __init__(self, n_F, h, w, n_gnn = 1, distance = "manhattan"):
        '''helper
            h, w: the height, width of the input features (bs, F, h, w)
            distance: manhattan or euler for generate distance-related adjacent matrix (default: manhattan)
        '''
        super().__init__()
        ## solve for spatial-related Adjacent matrix A
        N = int(h*w)
        self.A = nn.Parameter(torch.empty((N,N)), requires_grad=False)
        if distance=="manhattan":
            dist = lambda x, y: float(abs(x//w-y//w)+abs(x%w-y%w))
        elif distance=="euler":
            dist = lambda x, y: math.sqrt(float((x//w-y//w)**2+(x%w-y%w)**2))
        else:
            raise "unknow distance options!"
        for i in range(N):
            for j in range(i, N):
                self.A[i, j] = -dist(i,j)
                self.A[j, i] = -dist(j,i)
        ## n_gnn
        self.gnn = GNN(n_F=n_F, n_gnn=n_gnn)
    
    def forward(self, X, channels_last=True, reverse=False):
        ''' helper:
            channels_last: channels is in the last dim (dim=-1)?
            X: bs, F, N when channels_last=False
            X: bs, N, F when channels_last=True

            return X bs, N, F
        '''
        if channels_last==False:
            X = X.permute(0, 2, 1) ## -> bs, N, F
        return self.gnn(self.A, X, reverse)

class AttentionGNN(nn.Module):
    def __init__(self, n_F, n_gnn=1):
        super().__init__()
        self.gnn = GNN(n_F=n_F, n_gnn=n_gnn)

    def forward(self, Q, K, X, reverse=False):
        '''helper:
            Q: bs, N, k
            K: bs, N, k
            X: bs, N, k
        '''
        d = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(d)
        # scores = (scores + scores.transpose(-2,-1))/2.0
        return self.gnn(scores, X, reverse)

class AttenMultiHead(nn.Module):
    def __init__(self, k, h = 8):
        super().__init__()
        assert k%h==0
        self.h = h

        self.bnQ = LayerNorm(k)
        self.bnK = LayerNorm(k)
        self.bnV = LayerNorm(k)
        self.linearQ = nn.Linear(k, k)
        self.linearK = nn.Linear(k, k)
        self.linearV = nn.Linear(k, k)
        self.attenGNNs = nn.ModuleList([AttentionGNN(k//h) for _ in range(h)])
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, reverse=False):
        '''heads = self.h
        Q,K,V: bs, N, k
        RETURN bs, N, k
        '''
        block = Q.shape[-1]//self.h
        Qs = self.linearQ(self.bnQ(Q)).split(block, dim=-1)
        Ks = self.linearK(self.bnK(K)).split(block, dim=-1)
        Vs = self.linearV(self.bnV(V)).split(block, dim=-1)
        Vouts = torch.cat([
            self.attenGNNs[i](Qs[i], Ks[i], Vs[i], reverse) for i in range(self.h)
        ],dim=-1) ## bs, N, k
        V = self.bnV(V)+self.dropout(Vouts)
        return self.dropout(V)
