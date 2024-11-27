import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter

# Define the Spherical GNN Layer
class SphericalGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rbf=16, in_degree=0, r_degree=2, out_degree=2):
        super(SphericalGNNLayer, self).__init__(aggr='add')
        self.num_rbf = num_rbf
        self.in_degree = in_degree
        self.r_degree = r_degree
        self.out_degree = out_degree
        self.rbf_centers = torch.linspace(0, 5, num_rbf)  # Radial basis function centers
        self.rbf_gamma = torch.tensor(1.0)  # Gamma parameter for RBF

        
        self.W = nn.Parameter(torch.randn( self.in_degree+1, self.r_degree+1, self.out_degree+1, in_channels, out_channels, self.num_rbf))

        


    def associated_legendre_polynomials(L, x):
        """
        Compute the associated Legendre polynomials.

        Parameters:
        L (int): The maximum degree of the polynomials.
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: A tensor containing the associated Legendre polynomials.
        """
        P = [torch.ones_like(x) for _ in range((L+1)*L//2)]
        
        # Compute the polynomials for l in range(1, L)
        for l in range(1, L):
            P[(l+3)*l//2] = - np.sqrt((2*l-1)/(2*l)) * torch.sqrt(1-x**2) * P[(l+2)*(l-1)//2]
        
        # Compute the polynomials for m in range(L-1)
        for m in range(L-1):
            P[(m+2)*(m+1)//2+m] = x * np.sqrt(2*m+1) * P[(m+1)*m//2+m]
            for l in range(m+2, L):
                P[(l+1)*l//2+m] = ((2*l-1)*x*P[l*(l-1)//2 + m]/np.sqrt((l**2-m**2)) - P[(l-1)*(l-2)//2+m]*np.sqrt(((l-1)**2-m**2)/(l**2-m**2)))
        return torch.stack(P, dim=0)

    def spherical_harmonics(L, THETA, PHI, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Compute the spherical harmonics.

        Parameters:
        L (int): The maximum degree of the harmonics.
        THETA (torch.Tensor): The theta angles.
        PHI (torch.Tensor): The phi angles.
        device (torch.device): The device to use for computations (default is CUDA if available).

        Returns:
        list: A list of tensors containing the spherical harmonics for each degree l.
        """
        P = associated_legendre_polynomials(L, torch.cos(PHI))
        M2 =  [torch.zeros_like(THETA) for _ in range(2*(L-1)+1)]
        output =  [[torch.zeros_like(THETA, device = device) for _ in range(2*l+1)] for l in range(L)]
        
        # Compute cosine and sine components for each m
        for m in range(L):
            if m > 0:
                M2[L-1+m] = torch.cos(m*THETA)
                M2[L-1-m] = torch.sin(m*THETA)
            else:
                M2[L-1]  = torch.ones_like(THETA)
        
        # Compute the spherical harmonics for each l and m
        for l in range(L):
            for m in range(l+1):
                if m > 0:
                    output[l][l+m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1+m]
                    output[l][l-m] = np.sqrt(2)*P[(l+1)*l//2+m]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1-m]
                else:
                    output[l][l  ] = P[(l+1)*l//2]*np.sqrt((2*l+1)/(4*np.pi))*M2[L-1]
        
        return torch.concat([torch.stack(output_i, dim = 0).to(device) for output_i in output])

    def tensor_product(f_j, Y_r, cg, W, rbf, in_degree, r_degree, out_degree):
        # Tensor product using Clebsch-Gordan coefficients
        in_degree_to_order = torch.tensor([int(np.floor(np.sqrt(i + 1)))-1 for i in range((in_degree + 1) ** 2)], dtype=torch.long)
        r_degree_to_order = torch.tensor([int(np.floor(np.sqrt(i + 1)))-1 for i in range((r_degree + 1) ** 2)], dtype=torch.long)
        out_degree_to_order = torch.tensor([int(np.floor(np.sqrt(i + 1)))-1 for i in range((out_degree + 1) ** 2)], dtype=torch.long)
        # print(W.shape, in_degree_to_order.shape, r_degree_to_order.shape, out_degree_to_order.shape)
        W_spanned = ((W[in_degree_to_order])[:, r_degree_to_order])[:, :, out_degree_to_order]
        # print(cg.shape,  in_degree, r_degree, out_degree)
        # print( f_j.shape)
        # print(Y_r.shape)
        # print((cg[:(in_degree + 1) ** 2, :(r_degree + 1) ** 2, :(out_degree + 1) ** 2, ]).shape)
        # print(W_spanned.shape)
        # print(rbf.shape)
        out = torch.einsum('exa, ye, xyz, xyzabr, er->ezb', f_j, Y_r, cg[:(in_degree + 1) ** 2, :(r_degree + 1) ** 2, :(out_degree + 1) ** 2, ], W_spanned, rbf)
        return out


    def forward(self, x, pos, edge_index):
        # Compute pairwise distances
        row, col = edge_index
        diff = pos[row] - pos[col]
        dist = diff.norm(dim=-1)

        # Compute RBF and spherical harmonics
        rbf = torch.exp(-self.rbf_gamma[None] * (dist[:, None] - self.rbf_centers[None]) ** 2)
        sh = self.spherical_harmonics(diff)

        # Perform message passing
        # print(edge_index.shape, x.shape, rbf.shape, sh.shape)
        out = self.propagate(edge_index, x=x, rbf=rbf, sh=sh)
        return out

    def message(self, x_j, rbf, sh):
        # Tensor product of input features with spherical 
        x_j  = torch.reshape(x_j, (x_j.shape[0], (self.in_degree+1)**2, -1))
        tp = tensor_product(x_j, sh, self.cg, self.W, rbf, self.in_degree, self.r_degree, self.out_degree)
        return tp.reshape(tp.shape[0], -1)

    

    def spherical_harmonics(self, vectors):
        # Compute spherical harmonics of vectors
        theta = torch.atan2(vectors[:, 1], vectors[:, 0])
        phi = torch.acos(vectors[:, 2] / vectors.norm(dim=-1))
        sh = spherical_harmonics(self.r_degree + 1, theta, phi)
        return sh