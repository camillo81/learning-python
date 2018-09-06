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
            