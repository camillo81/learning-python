'''Basic classes and functions for finite element (FEM) programming.
    Written by Stefan Possanner
    stefan.possanner@ma.tum.de

   Contains:
       LagrangeShape : class
           The class for 1D Lagrange shape functions on the interval [-1,1].
       lag_assemb : function
           Computes the mass and stiffness matrices from Lagrange basis functions.
       L2prod_shape : function
           Computes the L2 scalar product of a given function with each element 
           of a basis defined from shape functions.
    
'''

import numpy as np
from scipy.integrate import fixed_quad as gauss_int

class LagrangeShape:
    '''The class for 1D Lagrange shape functions on the interval [-1,1].
    
    Parameters: 
        pts : ndarray
            1D array of increasing values in [-1, 1] defining the Lagrange polynomials.   
                    
    Returns:
        self.kind : string
            Is set to 'lagrange'.
        self.d : int
            Polynomial degree.
        self.s : ndarray
            The input array pts.
        self.eta : list
            List elements are the shape functions in 'poly1d' format.
        self.Deta : list
            List elements are the derivatives of the shape functions in 'poly1d' format.  
        self.mass : ndarray
            Mass matrix.  
        self.stiff : ndarray
            Stiffness matrix.
    '''
    
    kind = 'lagrange'
    
    def __init__(self, pts):
        
        # polynomial degree
        self.d = len(pts) - 1
        # elements of the reference interval
        self.s = pts
        # shape functions
        self.eta = [] 
        for i in range(self.d + 1):
            condition = self.s != self.s[i]
            roots = np.compress(condition, self.s) 
            self.eta.append(np.poly1d(roots, r=True)) # Numerator of Lagrange polynomial
            for j in range(len(roots)):
                self.eta[i] /= self.s[i] - roots[j] # Denominator of Lagrange polynomial
                
        # derivatives of shape functions
        self.Deta = []
        for i in range(self.d + 1):
            self.Deta.append(np.polyder(self.eta[i]))
                
        # mass and stiffness matrix:
        self.mass = np.zeros((self.d + 1, self.d + 1))
        self.stiff = np.zeros((self.d + 1, self.d + 1))
        for i in range(self.d + 1):
            for j in range(self.d + 1): 
                antider = np.polyint(self.eta[i]*self.eta[j])
                self.mass[i, j] = antider(1) - antider(-1)
                antider_D = np.polyint(self.Deta[i]*self.Deta[j])
                self.stiff[i, j] = antider_D(1) - antider_D(-1)
                
                
def lag_assemb(el_b, mass_eta, stiff_eta, bcs=2):
    ''' Computes the mass and stiffness matrices from Lagrange basis functions.
    
    Parameters:
        el_b : ndarray
            1D array of element interfaces from left to right including the boundaries. 
        mass_eta : ndarray
            The mass matrix (2D) of the shape functions defined on [-1, 1].
        stiff_eta : ndarray
            The stiffness matrix (2D) of the shape functions defined on [-1, 1].
        bcs : int
            Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
            1 stands for periodic.
            
    Returns:
        Nel : int
            The number of elements, Nel = len(el_b) - 1.
        mass : ndarray
            The mass matrix. If m = np.size(mass_eta[:, 0]) denotes the size of the local mass matrix, 
            then np.size(mass[:, 0]) = Nel*m - (Nel - 1) - bcs.
        stiff : ndarray
            The stiffness matrix. Same size as mass.
    '''
    
    Nel = len(el_b) - 1
    # number of elements
    
    m = mass_eta[:, 0].size 
    # size of local mass matrix (of the shape functions)
    
    d = m - 1
    # polynomial degree
    
    Ntot = Nel*m - (Nel - 1)
    NN = Ntot - bcs
    # number of degrees of freedom (Ntot including the boundary)
    
    mass = np.zeros((NN, NN))
    stiff = np.zeros((NN, NN))
    # initiate mass and stiffness matrix

    # left boundary:
    if bcs == 2:
        mass[:d, :d] = (el_b[1] - el_b[0])/2*mass_eta[1:, 1:]
        stiff[:d, :d] = 2/(el_b[1] - el_b[0])*stiff_eta[1:, 1:]
        index = d - 1
    else:
        print('Type of boundary condition not yet implemented, exiting ...')
        return

    # bulk:
    for i in np.arange(1, Nel - 1):
        mass[index:index + d + 1, index:index + d + 1] += (el_b[i + 1] - el_b[i])/2*mass_eta[:, :] 
        stiff[index:index + d + 1, index:index + d + 1] += 2/(el_b[i + 1] - el_b[i])*stiff_eta[:, :] 
        index += d
        # remark the '+=' in mass (stiff) for the cumulative sum for overlapping degrees of freedom

    # right boundary
    if bcs == 2:
        mass[index:index + d, index:index + d] += (el_b[-1] - el_b[-2])/2*mass_eta[:-1, :-1] 
        stiff[index:index + d, index:index + d] += 2/(el_b[-1] - el_b[-2])*stiff_eta[:-1, :-1] 
    else:
        print('Type of boundary condition not yet implemented, exiting ...')
        return
    
    return Nel, mass, stiff
            
    
def L2prod_shape():