
# https://github.com/AdrianArnaiz/DiffWire

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseGraphConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, DenseGraphConv
import numpy as np


class GAPNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=32, derivative=None, EPS=1e-15, device=None):
        super(GAPNet, self).__init__()
        self.device = device
        self.derivative = derivative
        self.EPS = EPS
        # GCN Layer - MLP - Dense GCN Layer
        #self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_of_centers2 = 16  # k2
        #num_of_centers2 =  10 # k2
        #num_of_centers2 =  5 # k2
        num_of_centers1 = 2  # k1 #Fiedler vector
        # The degree of the node belonging to any of the centers
        self.pool1 = Linear(hidden_channels, num_of_centers1)
        self.pool2 = Linear(hidden_channels, num_of_centers2)
        # MLPs towards out
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

        # Input: Batch of 20 graphs, each node F=3 features
        #        N1 + N2 + ... + N2 = 661
        # TSNE here?
    def forward(self, x, edge_index, batch):  # x torch.Size([N, N]),  data.batch  torch.Size([661])

        # Make all adjacencies of size NxN
        adj = to_dense_adj(edge_index, batch)  # adj torch.Size(B, N, N])
        #print("adj_size", adj.size())
        #print("adj",adj)

        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40:
        #print("x size", x.size())
        x, mask = to_dense_batch(x, batch)  # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20
        #print("x size", x.size())

        x = self.lin1(x)
        # First mincut pool for computing Fiedler adn rewire
        s1 = self.pool1(x)
        #s1 = torch.variable()#s1 torch.Size([20, N, k1=2)
        #s1 = Variable(torch.randn(D_in, H).type(float16), requires_grad=True)
        #print("s 1st pool",s1)
        #print("s 1st pool size", s1.size())

        if torch.isnan(adj).any():
            print("adj nan")
        if torch.isnan(x).any():
            print("x nan")

        # REWIRING
        #start = time.time()
        adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask, derivative=self.derivative, EPS=self.EPS, device=self.device)  # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])
        #print('\t\tdense_mincut_rewiring: {:.6f}s'.format(time.time()- start))
        #print("x",x)
        #print("adj",adj)
        #print("x and adj sizes", x.size(), adj.size())
        #adj = torch.softmax(adj, dim=-1)
        #print("adj softmaxed", adj)

        # CONV1: Now on x and rewired adj:
        x = self.conv1(x, adj)  #out: x torch.Size([20, N, F'=32])
        #print("x_1 ", x)
        #print("x_1 size", x.size())

        # MLP of k=16 outputs s
        #print("adj_size", adj.size())
        s2 = self.pool2(x)  # s torch.Size([20, N, k])
        #print("s 2nd pool", s2)
        #print("s 2nd pool size", s2.size())
        #adj = torch.softmax(adj, dim=-1)

        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        #x, adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask) # x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS)  # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        #print("lossses2",mincut_loss2, ortho_loss2)
        #print("mincut pool x", x)
        #print("mincut pool adj", adj)
        #print("mincut pool x size", x.size())
        #print("mincut pool adj size", adj.size()) # Some nan in adjacency: maybe comming from the rewiring-> dissapear after clipping

        # CONV2: Now on coarsened x and adj:
        x = self.conv2(x, adj)  #out x torch.Size([20, 16, 32])
        #print("x_2", x)
        #print("x_2 size", x.size())

        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1)  # x torch.Size([20, 32])
        #print("mean x_2 size", x.size())

        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x))  # x torch.Size([20, 32])
        #print("final x1 size", x.size())
        x = self.lin3(x)  #x torch.Size([20, 2])
        #print("final x2 size", x.size())
        #print("losses: ", mincut_loss1, mincut_loss2, ortho_loss2, mincut_loss2)
        mincut_loss = mincut_loss1 + mincut_loss2
        ortho_loss = ortho_loss1 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), mincut_loss, ortho_loss


class CTNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_centers, hidden_channels=32, EPS=1e-15):
        super(CTNet, self).__init__()
        self.EPS = EPS
        # GCN Layer - MLP - Dense GCN Layer
        #self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        # The degree of the node belonging to any of the centers
        num_of_centers1 = k_centers  # k1 #order of number of nodes
        self.pool1 = Linear(hidden_channels, num_of_centers1)
        num_of_centers2 = 16  # k2 #mincut
        self.pool2 = Linear(hidden_channels, num_of_centers2)

        # MLPs towards out
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):  # x torch.Size([N, N]),  data.batch  torch.Size([661])
        # Make all adjacencies of size NxN
        adj = to_dense_adj(edge_index, batch)  # adj torch.Size(B, N, N])
        #print("adj_size", adj.size())
        #print("adj",adj)

        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40:
        #print("x size", x.size())
        x, mask = to_dense_batch(x, batch)  # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20
        #print("x size", x.size())

        x = self.lin1(x)
        # First mincut pool for computing Fiedler adn rewire
        s1 = self.pool1(x)
        #s1 = torch.variable()#s1 torch.Size([20, N, k1=2)
        #s1 = Variable(torch.randn(D_in, H).type(float16), requires_grad=True)
        #print("s 1st pool",s1)
        #print("s 1st pool size", s1.size())

        if torch.isnan(adj).any():
            print("adj nan")
        if torch.isnan(x).any():
            print("x nan")

        # CT REWIRING
        adj, CT_loss, ortho_loss1 = dense_CT_rewiring(x, adj, s1, mask, EPS=self.EPS)  # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])

        #print("CT_loss, ortho_loss1", CT_loss, ortho_loss1)
        #print("x",x)
        #print("adj",adj)
        #print("x and adj sizes", x.size(), adj.size())
        #adj = torch.softmax(adj, dim=-1)
        #print("adj softmaxed", adj)

        # CONV1: Now on x and rewired adj:
        x = self.conv1(x, adj)  #out: x torch.Size([20, N, F'=32])
        #print("x_1 ", x)
        #print("x_1 size", x.size())

        # MLP of k=16 outputs s
        #print("adj_size", adj.size())
        s2 = self.pool2(x)  # s torch.Size([20, N, k])
        #print("s 2nd pool", s2)
        #print("s 2nd pool size", s2.size())
        #adj = torch.softmax(adj, dim=-1)

        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        #x, adj, mincut_loss1, ortho_loss1 = dense_mincut_rewiring(x, adj, s1, mask) # x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS)  # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])
        #print("lossses2",mincut_loss2, ortho_loss2)
        #print("mincut pool x", x)
        #print("mincut pool adj", adj)
        #print("mincut pool x size", x.size())
        #print("mincut pool adj size", adj.size()) # Some nan in adjacency: maybe comming from the rewiring-> dissapear after clipping

        # CONV2: Now on coarsened x and adj:
        x = self.conv2(x, adj)  #out x torch.Size([20, 16, 32])
        #print("x_2", x)
        #print("x_2 size", x.size())

        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1)  # x torch.Size([20, 32])
        #print("mean x_2 size", x.size())

        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x))  # x torch.Size([20, 32])
        #print("final x1 size", x.size())
        x = self.lin3(x)  #x torch.Size([20, 2])
        #print("final x2 size", x.size())
        CT_loss = CT_loss + ortho_loss1
        mincut_loss = mincut_loss2 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), CT_loss, mincut_loss


class MinCutNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=32, EPS=1e-15):
        super(MinCutNet, self).__init__()
        self.EPS = EPS
        # GCN Layer - MLP - Dense GCN Layer
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        # The degree of the node belonging to any of the centers
        num_of_centers2 = 16  # k2 #mincut
        self.pool2 = Linear(hidden_channels, num_of_centers2)

        # MLPs towards out
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):  # x torch.Size([N, N]),  data.batch  torch.Size([661])

        # Make all adjacencies of size NxN
        adj = to_dense_adj(edge_index, batch)  # adj torch.Size(B, N, N])
        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40:
        x, mask = to_dense_batch(x, batch)  # x torch.Size([20, N, 32]) ; mask torch.Size([20, N]) batch_size=20

        x = self.lin1(x)

        if torch.isnan(adj).any():
            print("adj nan")
        if torch.isnan(x).any():
            print("x nan")

        # CONV1: Now on x and rewired adj:
        x = self.conv1(x, adj)  #out: x torch.Size([20, N, F'=32])

        # MLP of k=16 outputs s
        s2 = self.pool2(x)  # s torch.Size([20, N, k])

        # MINCUT_POOL
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        x, adj, mincut_loss2, ortho_loss2 = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS)  # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])

        # CONV2: Now on coarsened x and adj:
        x = self.conv2(x, adj)  #out x torch.Size([20, 16, 32])

        # Readout for each of the 20 graphs
        #x = x.mean(dim=1) # x torch.Size([20, 32])
        x = x.sum(dim=1)  # x torch.Size([20, 32])
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x))  # x torch.Size([20, 32])
        x = self.lin3(x)  #x torch.Size([20, 2])

        mincut_loss = mincut_loss2 + ortho_loss2
        #print("x", x)
        return F.log_softmax(x, dim=-1), mincut_loss2, ortho_loss2


# Commute Times rewiring
# Graph Convolutional Network layer where the graph structure is given by an adjacency matrix.
# We recommend user to use this module when applying graph convolution on dense graphs.
# from torch_geometric.nn import GCNConv, DenseGraphConv
# import torch
# from layers.utils.ein_utils import _rank3_diag, _rank3_trace


#  Trace of a tensor [1,k,k]
def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


# Diagonal version of a tensor [1,n] -> [1,n,n]
def _rank3_diag(x):
    # Eye matrix of n=x.size(1): [n,n]
    eye = torch.eye(x.size(1)).type_as(x)
    #print(eye.size())
    #print(x.unsqueeze(2).size())
    # x.unsqueeze(2) adds a second dimension to [1,n] -> [1,n,1]
    # expand(*x.size(), x.size(1)) takes [1,n,1] and expands [1,n] with n -> [1,n,n]
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


def approximate_Fiedler(s, device=None):  # torch.Size([20, N, k]) One k for each N of each graph (asume k=2)
    """
  Calculate approximate fiedler vector from S matrix. S in R^{B x N x 2} and fiedler vector S in R^{B x N}
  """
    s_0 = s.size(0)  #number of graphs
    s_1 = s.size(1)
    maxcluster = torch.argmax(s, dim=2)  # torch.Size([20, N]) with binary values {0,1} if k=2
    # trimmed_s = torch.ones(s_0, s_1).to(device)
    trimmed_s = torch.ones(s_0, s_1, dtype=torch.float).to(device)
    # print("maxcluster.shape", maxcluster.shape)
    # print("trimmed_s.shape", trimmed_s.shape)
    #print('\t'*4,'[DEVICES] s device', s.device,' -- trimmed_s device', trimmed_s.device,' -- maxcluster device', trimmed_s.device)
    trimmed_s[maxcluster == 1] = -1 / np.sqrt(float(s_1))
    trimmed_s[maxcluster == 0] = 1 / np.sqrt(float(s_1))
    return trimmed_s


def NLderivative_of_lambda2_wrt_adjacency(adj, d_flat, fiedlers, EPS, device):  # fiedlers torch.Size([20, N])
    """
  Complex derivative

  Args:
      adj (_type_): _description_
      d_flat (_type_): _description_
      fiedlers (_type_): _description_

  Returns:
      _type_: _description_
  """
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    # Batched structures for the complex derivative
    d_flat2 = torch.sqrt(d_flat + EPS)[:, None] + EPS  # d torch.Size([B, 1, N])
    #print("first d_flat2 size", d_flat2.size())
    Ahat = (adj / d_flat2.transpose(1, 2))  # [B, N, N] / [B, N, 1] -> [B, N, N]
    AhatT = (adj.transpose(1, 2) / d_flat2.transpose(1, 2))  # [B, N, N] / [B, N, 1] -> [B, N, N]
    dinv = 1 / (d_flat + EPS)[:, None]
    dder = -0.5 * dinv * d_flat2
    dder = dder.transpose(1, 2)  # [B, N, 1]
    # Storage
    # derivatives = torch.ones(B, N, N).to(device)
    derivatives = torch.ones(B, N, N, dtype=torch.float).to(device)

    for b in range(B):
        # Eigenvectors
        u2 = fiedlers[b, :]
        u2 = u2.unsqueeze(1)  # [N, 1]
        #u2 = u2.to(device) #its already in device because fiedlers is already in device
        #print("size of u2", u2.size())

        # First term central: [N,1]x ([1,N]x[N,N]x[N,1]) x [N,1]
        firstT = torch.matmul(torch.matmul(u2.T, AhatT[b, :, :]), u2)  # [1,N]x[N,N]x[N,1] -> [1]
        #print("first term central size", firstT.size())
        firstT = torch.matmul(torch.matmul(dder[b, :], firstT), torch.ones(N).unsqueeze(0).to(device))

        # Second term
        secT = torch.matmul(torch.matmul(u2.T, Ahat[b, :, :]), u2)  # [1,N]x[N,N]x[N,1] -> [1]
        #print("second term central size", secT.size())
        secT = torch.matmul(torch.matmul(dder[b, :], secT), torch.ones(N).unsqueeze(0).to(device))

        # Third term
        u2u2T = torch.matmul(u2, u2.T)  # [N,1] x [1,N] -> [N,N]
        #print("u2u2T size", u2u2T.size())
        #print("d_flat2[b,:] size", d_flat2[b,:].size())
        Du2u2TD = (u2u2T / d_flat2[b, :]) / d_flat2[b, :].transpose(0, 1)
        #print("size of Du2u2TD", Du2u2TD.size())
        # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL
        #dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)),torch.ones(N,N)) - u2u2T
        dl2 = firstT + secT + Du2u2TD
        # Symmetrize and subtract the diag since it is an undirected graph
        #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
        derivatives[b, :, :] = -dl2
    return derivatives  # derivatives torch.Size([20, N, N])


def NLfiedler_values(L, d_flat, fiedlers, EPS, device):  # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    #print("original fiedlers size", fiedlers.size())

    # Batched Fiedlers
    d_flat2 = torch.sqrt(d_flat + EPS)[:, None] + EPS  # d torch.Size([B, 1, N])
    #print("d_flat2 size", d_flat2.size())
    fiedlers = fiedlers.unsqueeze(2)  # [B, N, 1]
    #print("fiedlers size", fiedlers.size())
    fiedlers_hats = (fiedlers / d_flat2.transpose(1, 2))  # [B, N, 1] / [B, N, 1] -> [B, N, 1]
    gfiedlers_hats = (fiedlers * d_flat2.transpose(1, 2))  # [B, N, 1] * [B, N, 1] -> [B, N, 1]
    #print("fiedlers size", fiedlers_hats.size())
    #print("gfiedlers size", gfiedlers_hats.size())

    #Laplacians = torch.ones(B, N, N)
    # fiedler_values = torch.ones(B).to(device)
    fiedler_values = torch.ones(B, dtype=torch.float).to(device)
    for b in range(B):
        f = fiedlers_hats[b, :]
        g = gfiedlers_hats[b, :]
        num = torch.matmul(f.T, torch.matmul(L[b, :, :], f))  # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
        den = torch.matmul(g.T, g)
        #print("num fied", num.size())
        #print("den fied", den.size())
        #print("g size", g.size())
        #print("f size", f.size())
        #print("L size", L[b,:,:].size())
        fiedler_values[b] = N * torch.abs(num / (den + EPS))
    return fiedler_values  # torch.Size([B])


def derivative_of_lambda2_wrt_adjacency(fiedlers, device):  # fiedlers torch.Size([20, N])
    """
  Simple derivative
  """
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    derivatives = torch.ones(B, N, N, dtype=torch.float).to(device)
    for b in range(B):
        u2 = fiedlers[b, :]
        u2 = u2.unsqueeze(1)
        #print("size of u2", u2.size())
        u2u2T = torch.matmul(u2, u2.T)
        #print("size of u2u2T", u2u2T.size())
        # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL
        dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)), torch.ones(N, N, dtype=torch.float).to(device)) - u2u2T
        # Symmetrize and subtract the diag since it is an undirected graph
        #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
        derivatives[b, :, :] = dl2

    return derivatives  # derivatives torch.Size([20, N, N])


def fiedler_values(adj, fiedlers, EPS, device):  # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    #Laplacians = torch.ones(B, N, N)
    fiedler_values = torch.ones(B, dtype=torch.float).to(device)
    for b in range(B):
        # Compute un-normalized Laplacian
        A = adj[b, :, :]
        D = A.sum(dim=1)
        D = torch.diag(D)
        L = D - A
        #Laplacians[b,:,:] = L
        #if torch.min(A)<0:
        #  print("Negative adj")
        # Compute numerator
        f = fiedlers[b, :].unsqueeze(1)
        #f = f.to(device)
        num = torch.matmul(f.T, torch.matmul(L, f))  # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
        # Create complete graph Laplacian
        CA = torch.ones(N, N, dtype=torch.float).to(device) - torch.eye(N).to(device)
        CD = CA.sum(dim=1)
        CD = torch.diag(CD)
        CL = CD - CA
        CL = CL.to(device)
        # Compute denominator
        den = torch.matmul(f.T, torch.matmul(CL, f))
        fiedler_values[b] = N * torch.abs(num / (den + EPS))

    return fiedler_values  # torch.Size([B])


def NLderivative_of_lambda2_wrt_adjacencyV2(adj, d_flat, fiedlers, EPS, device):  # fiedlers torch.Size([20, N])
    """
    Complex derivative
    Args:
      adj (_type_): _description_
      d_flat (_type_): _description_
      fiedlers (_type_): _description_
    Returns:
      _type_: _description_
    """
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    # Batched structures for the complex derivative
    d_flat2 = torch.sqrt(d_flat + EPS)[:, None] + EPS  # d torch.Size([B, 1, N])
    d_flat = d_flat2.squeeze(1)
    #print("first d_flat2 size", d_flat2.size())
    d_half = _rank3_diag(d_flat)  # d torch.Size([B, N, N])
    #print("d size", d.size())
    Ahat = (adj / d_flat2.transpose(1, 2))  # [B, N, N] / [B, N, 1] -> [B, N, N]
    AhatT = (adj.transpose(1, 2) / d_flat2.transpose(1, 2))  # [B, N, N] / [B, N, 1] -> [B, N, N]
    dinv = 1 / (d_flat + EPS)[:, None]
    dder = -0.5 * dinv * d_flat2
    dder = dder.transpose(1, 2)  # [B, N, 1]
    # Storage
    derivatives = torch.ones(B, N, N, dtype=torch.float).to(device)
    for b in range(B):
        # Eigenvectors
        u2 = fiedlers[b, :]
        u2 = u2.unsqueeze(1)  # [N, 1]
        #u2 = u2.to(device) #its already in device because fiedlers is already in device
        #print("size of u2", u2.size())
        # First term central: [N,1]x ([1,N]x[N,N]x[N,1]) x [N,1]
        firstT = torch.matmul(torch.matmul(u2.T, torch.matmul(d_half[b, :, :], AhatT[b, :, :])), u2)  # [1,N]x[N,N]x[N,1] -> [1]
        #print("first term central size", firstT.size())
        firstT = torch.matmul(torch.matmul(dder[b, :], firstT), torch.ones(N, dtype=torch.float).unsqueeze(0).to(device))
        #print("first term  size", firstT.size())
        # Second term
        secT = torch.matmul(torch.matmul(u2.T, torch.matmul(d_half[b, :, :], Ahat[b, :, :])), u2)  # [1,N]x[N,N]x[N,1] -> [1]
        #print("second term central size", secT.size())
        secT = torch.matmul(torch.matmul(dder[b, :], secT), torch.ones(N, dtype=torch.float).unsqueeze(0).to(device))
        # Third term
        Du2u2TD = torch.matmul(u2, u2.T)  # [N,1] x [1,N] -> [N,N]
        #print("Du2u2T size", u2u2T.size())
        #print("d_flat2[b,:] size", d_flat2[b,:].size())
        #Du2u2TD = (u2u2T / d_flat2[b,:]) / d_flat2[b,:].transpose(0, 1)
        #print("size of Du2u2TD", Du2u2TD.size())
        # dl2 = torch.matmul(torch.diag(u2u2T),torch.ones(N,N)) - u2u2T ERROR FUNCTIONAL
        #dl2 = torch.matmul(torch.diag(torch.diag(u2u2T)),torch.ones(N,N)) - u2u2T
        dl2 = firstT + secT + Du2u2TD
        # Symmetrize and subtract the diag since it is an undirected graph
        #dl2 = dl2 + dl2.T - torch.diag(torch.diag(dl2))
        derivatives[b, :, :] = -dl2
    return derivatives  # derivatives torch.Size([20, N, N])


def NLfiedler_valuesV2(L, d, fiedlers, EPS, device):  # adj torch.Size([B, N, N]) fiedlers torch.Size([B, N])
    N = fiedlers.size(1)
    B = fiedlers.size(0)
    #print("original fiedlers size", fiedlers.size())
    #print("d size", d.size())
    #Laplacians = torch.ones(B, N, N)
    fiedler_values = torch.ones(B, dtype=torch.float).to(device)
    for b in range(B):
        f = fiedlers[b, :].unsqueeze(1)
        num = torch.matmul(f.T, torch.matmul(L[b, :, :], f))  # f is [N,1], L is [N, N], f.T is [1,N] -> Lf is [N,1] -> f.TLf is [1]
        den = torch.matmul(f.T, torch.matmul(d[b, :, :], f))  # d is [N, N], f is [N,1], f.T is [1,N] -> [1,N] x [N, N] x [N, 1] is [1]
        fiedler_values[b] = N * torch.abs(num / (den + EPS))
        """print(f.shape)
    print(f.T.shape)
    print(num.shape, num)
    print(den.shape, den)
    print((N*torch.abs(num/(den + EPS))).shape)
    exit()"""
    return fiedler_values  # torch.Size([B])


def dense_CT_rewiring(x, adj, s, mask=None, EPS=1e-15):  # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20
    #print("Input x size to mincut pool", x.size())
    x = x.unsqueeze(0) if x.dim() == 2 else x  # x torch.Size([20, 40, 32]) if x has not 2 parameters
    #print("Unsqueezed x size to mincut pool", x.size(), x.dim()) # x.dim() is usually 3

    # adj torch.Size([20, N, N]) N=Mmax
    #print("Input adj size to mincut pool", adj.size())
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # adj torch.Size([20, N, N]) N=Mmax
    #print("Unsqueezed adj size to mincut pool", adj.size(), adj.dim()) # adj.dim() is usually 3

    # s torch.Size([20, N, k])
    s = s.unsqueeze(0) if s.dim() == 2 else s  # s torch.Size([20, N, k])
    #print("Unsqueezed s size", s.size())

    # x torch.Size([20, N, 32]) if x has not 2 parameters
    (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    #print("batch_size and num_nodes", batch_size, num_nodes, k) # batch_size = 20, num_nodes = N, k = 16
    s = torch.tanh(s)  # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())

    if mask is not None:  # NOT None for now
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        #print("mask size", mask.size()) # [20, N, 1]
        # Mask pointwise product. Since x is [20, N, 32] and s is [20, N, k]
        x, s = x * mask, s * mask  # x*mask = [20, N, 32]*[20, N, 1] = [20, N, 32] s*mask = [20, N, k]*[20, N, 1] = [20, N, k]
        #print("x and s sizes after multiplying by mask", x.size(), s.size()

    # CT regularization
    # Calculate degree d_flat and degree matrix d
    d_flat = torch.einsum('ijk->ij', adj)  # torch.Size([20, N])
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat) + EPS  # d torch.Size([20, N, N])
    #print("d size", d.size())

    # Calculate Laplacian L = D - A
    L = d - adj
    #print("Laplacian", L[1,:,:])

    # Calculate out_adj as A_CT = S.T*L*S
    # out_adj: this tensor contains A_CT = S.T*L*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), L), s)  #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal

    # Calculate CT_num
    CT_num = _rank3_trace(out_adj)  # mincut_num torch.Size([20]) one sum over each graph
    #print("CT_num size", CT_num.size())
    #print("CT_num", CT_num)
    # Calculate CT_den
    CT_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s)) + EPS  # [20, k, N]*[20, N, N]->[20, k, N]*[20, N, k] -> [20] one sum over each graph
    #print("CT_den size", CT_den.size())
    #print("CT_den", CT_den)

    # Calculate CT_dist (distance matrix)
    CT_dist = torch.cdist(s, s)  # [20, N, k], [20, N, k]-> [20,N,N]
    #print("CT_dist",CT_dist)

    # Calculate Vol (volumes): one per graph
    vol = _rank3_trace(d)  # torch.Size([20])
    #print("vol size", vol.size())

    #print("vol_flat size", vol_flat.size())
    vol = _rank3_trace(d)  # d torch.Size([20, N, N])
    #print("vol size", vol.size())
    #print("vol", vol)

    # Calculate out_adj as CT_dist*(N-1)/vol(G)
    N = adj.size(1)
    #CT_dist = (CT_dist*(N-1)) / vol.unsqueeze(1).unsqueeze(1)
    CT_dist = (CT_dist) / vol.unsqueeze(1).unsqueeze(1)
    #CT_dist = (CT_dist) / ((N-1)*vol).unsqueeze(1).unsqueeze(1)

    #print("R_dist",CT_dist)

    # Mask with adjacency if proceeds
    adj = CT_dist * adj
    #adj = CT_dist

    # Losses
    CT_loss = CT_num / CT_den
    CT_loss = torch.mean(CT_loss)  # Mean over 20 graphs!
    #print("CT_loss", CT_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  #[20, k, N]*[20, N, k]-> [20, k, k]
    #print("ss size", ss.size())
    i_s = torch.eye(k).type_as(ss)  # [k, k]
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s)
    #print("ortho_loss size", ortho_loss.size()) # [20] one sum over each graph
    ortho_loss = torch.mean(ortho_loss)
    #print("ortho_loss", ortho_loss)

    return adj, CT_loss, ortho_loss  # [20, k, 32], [20, B, N], [1], [1]


def dense_mincut_rewiring(x, adj, s, mask=None, derivative=None, EPS=1e-15, device=None):  # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20

    k = 2  #We want bipartition to compute spectral gap
    # adj torch.Size([20, N, N]) N=Mmax
    #print("Input adj size to mincut pool", adj.size())
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # adj torch.Size([20, N, N]) N=Mmax
    #print("Unsqueezed adj size to mincut pool", adj.size(), adj.dim()) # adj.dim() is usually 3

    # s torch.Size([20, N, k])
    s = s.unsqueeze(0) if s.dim() == 2 else s  #s torch.Size([20, N, k])
    #print("Unsqueezed s size", s.size())

    s = torch.softmax(s, dim=-1)  # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())
    #print("s softmax", s[0,1,:], torch.argmax(s,dim=(2)).size())

    # Put here the calculus of the degree matrix to optimize the complex derivative
    d_flat = torch.einsum('ijk->ij', adj)  # torch.Size([20, N])
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat)  # d torch.Size([20, N, N])
    #print("d size", d.size())

    # Batched Laplacian
    L = d - adj

    # REWIRING: UPDATING adj wrt s using derivatives -------------------------------------------------
    # Approximating the Fiedler vectors from s (assuming k=2)
    fiedlers = approximate_Fiedler(s, device)
    #print("fiedlers size", fiedlers.size())
    #print("fiedlers ", fiedlers)

    # Recalculate
    if derivative == "laplacian":
        der = derivative_of_lambda2_wrt_adjacency(fiedlers, device)
        fvalues = fiedler_values(adj, fiedlers, EPS, device)
    elif derivative == "normalized":
        #start = time.time()
        der = NLderivative_of_lambda2_wrt_adjacency(adj, d_flat, fiedlers, EPS, device)
        fvalues = NLfiedler_values(L, d_flat, fiedlers, EPS, device)
        #print('\t\t NLderivative_of_lambda2_wrt_adjacency: {:.6f}s'.format(time.time()- start))
    elif derivative == "normalizedv2":
        der = NLderivative_of_lambda2_wrt_adjacencyV2(adj, d_flat, fiedlers, EPS, device)
        fvalues = NLfiedler_valuesV2(L, d, fiedlers, EPS, device)

    mu = 0.01
    lambdaReg = 0.1
    lambdaReg = 1.0
    lambdaReg = 1.5
    lambdaReg = 2.0
    lambdaReg = 2.5
    lambdaReg = 5.0
    lambdaReg = 3.0
    lambdaReg = 1.0
    lambdaReg = 2.0
    #lambdaReg = 20.0

    Ac = adj.clone()
    for _ in range(5):
        #fvalues = fiedler_values(Ac, fiedlers)
        #print("Ac size", Ac.size())
        partialJ = 2 * (Ac - adj) + 2 * lambdaReg * der * fvalues.unsqueeze(1).unsqueeze(2)  # favalues is [B], partialJ is [B, N, N]
        #print("partialJ size", partialJ.size())
        #print("diag size", torch.diag_embed(torch.diagonal(partialJ,dim1=1,dim2=2)).size())
        dJ = partialJ + torch.transpose(partialJ, 1, 2) - torch.diag_embed(torch.diagonal(partialJ, dim1=1, dim2=2))
        # Update adjacency
        Ac = Ac - mu * dJ
        # Clipping: negatives to 0, positives to 1
        #print("Ac is", Ac, Ac.size())
        #Ac = torch.clamp(Ac, min=0.0, max=1.0)

        #print("Esta es la antigua adj",adj)
        #print("Esta es la antigua Ac",Ac)

        #print("Despues mask Ac",Ac)
        #print("Despues mask Adj",adj)
        #print("Mayores que 0",(Ac>0).sum()) #20,16,40
        #print("Menores que 0",(Ac<=0).sum())
        Ac = torch.softmax(Ac, dim=-1)
        Ac = Ac * adj
    #print("Min Fiedlers",min(fvalues))
    #print("NUeva salida",Ac)

    # out_adj: this tensor contains Apool = S.T*A*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)  # mincut_num torch.Size([20]) one sum over each graph
    #print("mincut_num size", mincut_num.size())
    #d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N])
    #print("d_flat size", d_flat.size())
    #d = _rank3_diag(d_flat) # d torch.Size([20, N, N])
    #print("d size", d.size())
    mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))  # [20, k, N]*[20, N, N]->[20, k, N]*[20, N, k] -> [20] one sum over each graph
    #print("mincut_den size", mincut_den.size())

    mincut_loss = -(mincut_num / mincut_den)
    #print("mincut_loss", mincut_loss)
    mincut_loss = torch.mean(mincut_loss)  # Mean over 20 graphs!

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  #[20, k, N]*[20, N, k]-> [20, k, k]
    #print("ss size", ss.size())
    i_s = torch.eye(k).type_as(ss)  # [k, k]
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    #print("ortho_loss size", ortho_loss.size()) # [20] one sum over each graph
    ortho_loss = torch.mean(ortho_loss)
    """# Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device) # range e.g. from 0 to 15 (k=16)
    # out_adj is [20, k, k]
    out_adj[:, ind, ind] = 0 # [20, k, k]  the diagnonal will be 0 now: Ahat = Apool - I_k*diag(Apool) (Eq. 8)
    #print("out_adj", out_adj[0,])

    # Final degree matrix and normalization of out_adj: Ahatpool = Dhat^{-1/2}AhatD^{-1/2} (Eq. 8)
    d = torch.einsum('ijk->ij', out_adj) #d torch.Size([20, k])
    #print("d size", d.size())
    d = torch.sqrt(d)[:, None] + EP S # d torch.Size([20, 1, k])
    #print("sqrt(d) size", d.size())
    #print( (out_adj / d).shape)  # out_adj is [20, k, k] and d is [20, 1, k] -> torch.Size([20, k, k]) 
    out_adj = (out_adj / d) / d.transpose(1, 2) # ([20, k, k] / [20, k, 1] ) -> [20, k, k]
    # out_adj torch.Size([20, k, k]) 
    #print("out_adj size", out_adj.size())"""
    return Ac, mincut_loss, ortho_loss  # [20, k, 32], [20, k, k], [1], [1]
    #return out, out_adj, mincut_loss, ortho_loss # [20, k, 32], [20, k, k], [1], [1]


def dense_mincut_pool(x, adj, s, mask=None, EPS=1e-15):  # x torch.Size([20, 40, 32]) ; mask torch.Size([20, 40]) batch_size=20
    #print("Input x size to mincut pool", x.size())
    x = x.unsqueeze(0) if x.dim() == 2 else x  # x torch.Size([20, 40, 32]) if x has not 2 parameters
    #print("Unsqueezed x size to mincut pool", x.size(), x.dim()) # x.dim() is usually 3

    # adj torch.Size([20, N, N]) N=Mmax
    #print("Input adj size to mincut pool", adj.size())
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # adj torch.Size([20, N, N]) N=Mmax
    #print("Unsqueezed adj size to mincut pool", adj.size(), adj.dim()) # adj.dim() is usually 3

    # s torch.Size([20, N, k])
    s = s.unsqueeze(0) if s.dim() == 2 else s  # s torch.Size([20, N, k])
    #print("Unsqueezed s size", s.size())

    # x torch.Size([20, N, 32]) if x has not 2 parameters
    (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    #print("batch_size and num_nodes", batch_size, num_nodes, k) # batch_size = 20, num_nodes = N, k = 16
    s = torch.softmax(s, dim=-1)  # torch.Size([20, N, k]) One k for each N of each graph
    #print("s softmax size", s.size())

    if mask is not None:  # NOT None for now
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        #print("mask size", mask.size()) # [20, N, 1]
        # Mask pointwise product. Since x is [20, N, 32] and s is [20, N, k]
        x, s = x * mask, s * mask  # x*mask = [20, N, 32]*[20, N, 1] = [20, N, 32] s*mask = [20, N, k]*[20, N, 1] = [20, N, k]
        #print("x and s sizes after multiplying by mask", x.size(), s.size())

    # out: this tensor contains Xpool=S.T*X (Eq. 7)
    out = torch.matmul(s.transpose(1, 2), x)  # [20, k, N] * [20, N, 32] will yield [20, k, 32]
    #print("out size", out.size())
    # out_adj: this tensor contains Apool = S.T*A*S so that we can take its trace and retain coarsened adjacency (Eq. 7)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  #[20, k, N]*[20, N, N]-> [20, k ,N]*[20, N, k] = [20, k, k] 20 graphs of k nodes
    #print("out_adj size", out_adj.size())
    #print("out_adj ", out_adj[0,]) # Has no zeros in the diagonal

    # MinCUT regularization.
    mincut_num = _rank3_trace(out_adj)  # mincut_num torch.Size([20]) one sum over each graph
    #print("mincut_num size", mincut_num.size())
    d_flat = torch.einsum('ijk->ij', adj) + EPS  # torch.Size([20, N])
    #print("d_flat size", d_flat.size())
    d = _rank3_diag(d_flat)  # d torch.Size([20, N, N])
    #print("d size", d.size())
    mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))  # [20, k, N]*[20, N, N]->[20, k, N]*[20, N, k] -> [20] one sum over each graph
    #print("mincut_den size", mincut_den.size())

    mincut_loss = -(mincut_num / mincut_den)
    #print("mincut_loss", mincut_loss)
    mincut_loss = torch.mean(mincut_loss)  # Mean over 20 graphs!

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  #[20, k, N]*[20, N, k]-> [20, k, k]
    #print("ss size", ss.size())
    i_s = torch.eye(k).type_as(ss)  # [k, k]
    ortho_loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    #print("ortho_loss size", ortho_loss.size()) # [20] one sum over each graph
    ortho_loss = torch.mean(ortho_loss)

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)  # range e.g. from 0 to 15 (k=16)
    # out_adj is [20, k, k]
    out_adj[:, ind, ind] = 0  # [20, k, k]  the diagnonal will be 0 now: Ahat = Apool - I_k*diag(Apool) (Eq. 8)
    #print("out_adj", out_adj[0,])

    # Final degree matrix and normalization of out_adj: Ahatpool = Dhat^{-1/2}AhatD^{-1/2} (Eq. 8)
    d = torch.einsum('ijk->ij', out_adj)  #d torch.Size([20, k])
    #print("d size", d.size())
    d = torch.sqrt(d + EPS)[:, None] + EPS  # d torch.Size([20, 1, k])
    #print("sqrt(d) size", d.size())
    #print( (out_adj / d).shape)  # out_adj is [20, k, k] and d is [20, 1, k] -> torch.Size([20, k, k])
    out_adj = (out_adj / d) / d.transpose(1, 2)  # ([20, k, k] / [20, k, 1] ) -> [20, k, k]
    # out_adj torch.Size([20, k, k])
    #print("out_adj size", out_adj.size())
    return out, out_adj, mincut_loss, ortho_loss  # [20, k, 32], [20, k, k], [1], [1]


class DiffWire(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        k_centers,
        derivative="normalized",
        hidden_channels=32,
        EPS=1e-15,  #=1e-10
        device=None,
    ):
        super().__init__()

        self.EPS = EPS
        self.derivative = derivative
        self.device = device

        # First X transformation
        self.lin1 = Linear(in_channels, hidden_channels)

        #Fiedler vector -- Pool previous to GAP-Layer
        self.pool_rw = Linear(hidden_channels, 2)

        #CT Embedding -- Pool previous to CT-Layer
        self.num_of_centers1 = k_centers  # k1 - order of number of nodes
        self.pool_ct = Linear(hidden_channels, self.num_of_centers1)  #CT

        #Conv1
        self.conv1 = DenseGraphConv(hidden_channels, hidden_channels)

        #MinCutPooling
        self.pool_mc = Linear(hidden_channels, 16)  #MC

        #Conv2
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        # MLPs towards out
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Make all adjacencies of size NxN
        adj = to_dense_adj(edge_index, batch)
        # Make all x_i of size N=MAX(N1,...,N20), e.g. N=40:
        x, mask = to_dense_batch(x, batch)

        x = self.lin1(x)

        if torch.isnan(adj).any():
            print("adj nan")
        if torch.isnan(x).any():
            print("x nan")

        #Gap Layer RW
        s0 = self.pool_rw(x)
        adj, mincut_loss_rw, ortho_loss_rw = dense_mincut_rewiring(x, adj, s0, mask, derivative=self.derivative, EPS=self.EPS, device=self.device)

        # CT REWIRING
        # First mincut pool for computing Fiedler adn rewire
        s1 = self.pool_ct(x)
        adj, CT_loss, ortho_loss_ct = dense_CT_rewiring(x, adj, s1, mask, EPS=self.EPS)  # out: x torch.Size([20, N, F'=32]),  adj torch.Size([20, N, N])

        # CONV1: Now on x and rewired adj:
        x = self.conv1(x, adj)

        # MINCUT_POOL
        # MLP of k=16 outputs s
        s2 = self.pool_mc(x)
        # Call to dense_cut_mincut_pool to get coarsened x, adj and the losses: k=16
        x, adj, mincut_loss, ortho_loss_mc = dense_mincut_pool(x, adj, s2, mask, EPS=self.EPS)  # out x torch.Size([20, k=16, F'=32]),  adj torch.Size([20, k2=16, k2=16])

        # CONV2: Now on coarsened x and adj:
        x = self.conv2(x, adj)

        # Readout for each of the 20 graphs
        x = x.sum(dim=1)
        # Final MLP for graph classification: hidden channels = 32
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        main_loss = mincut_loss_rw + CT_loss + mincut_loss
        ortho_loss = ortho_loss_rw + ortho_loss_ct + ortho_loss_mc
        #ortho_loss_rw/2 + (1/self.num_of_centers1)*ortho_loss_ct + ortho_loss_mc/16
        #print("x", x)
        return F.log_softmax(x, dim=-1), main_loss, ortho_loss
