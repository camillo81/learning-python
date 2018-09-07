'''Basic classes and functions for finite element (FEM) programming.
    Written by Stefan Possanner
    stefan.possanner@ma.tum.de

   Contains:
       LagrangeShape : class
           The class for 1D Lagrange shape functions on the interval [-1,1].
       lag_assemb : function
           Computes the mass and stiffness matrices from Lagrange basis functions.
       lag_L2prod : function
           Computes the L2 scalar product of a given function with each element 
           of a basis defined from shape functions.
       lag_fun : function
           Given a coefficient vector, returns a function in the space spanned by a Lagrange basis.
    
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
    for i in range(1, Nel - 1):
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
            
    
def lag_L2prod(fun, eta, el_b, bcs=2):
    '''Computes the L2 scalar product of a function with each element of a Lagrange basis
    defined from shape functions.
    
    Parameters:
        fun : function
            The input function, for example a 'lambda'-function.
        eta : list
            List elements are the shape functions in 'poly1d' format.
        el_b : ndarray
            1D array of element interfaces from left to right including the boundaries. 
        bcs : int
            Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
            1 stands for periodic.
            
    Returns:
        Nel : int
            The number of elements, Nel = len(el_b) - 1. 
        funbar : ndarray
            1D array of scalar products of fun with the basis functions.
        Nbase : int
            The number of basis functions, Nbase = np.size(fun_bar)
    '''
    
    from scipy.integrate import fixed_quad 
    
    Nel = len(el_b) - 1
    # number of elements
    
    m = len(eta)
    # number of shape functions
    
    d = m - 1
    # polynomial degree
    
    Ntot = Nel*m - (Nel - 1)
    Nbase = Ntot - bcs
    # number of basis functions
    
    funbar = np.zeros(Nbase)
    # initiate output vector
    
    index = 0
    # index of the basis function
    
    # left boundary:
    i = 0
    for j in range(1, m):
            
        fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
        # function fun transformed to the reference element [-1, 1]
        fun2 = lambda s: np.polyval(eta[j], s)
        # shape function
        
        fun12 = lambda s: fun1(s)*fun2(s)
        intval, foo = fixed_quad(fun12, -1, 1)
        funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
        # integral
        
        if j != d:
            index += 1
        # If it is the last shape function (j = d), the index rests the same
        # and the subsequent integral is added at the same position in funbar.
    
    # bulk:
    for i in range(1, Nel - 1): 
        for j in range(m):
            
            fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
            # function fun transformed to the reference element [-1, 1]
            fun2 = lambda s: np.polyval(eta[j], s)
            # shape function
            
            fun12 = lambda s: fun1(s)*fun2(s)
            intval, foo = fixed_quad(fun12, -1, 1)
            funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
            # integral
            if j != d:
                index += 1
        
    # right boundary:
    i = Nel - 1
    for j in range(d):
            
        fun1 = lambda s: fun( el_b[i] + (s + 1)/2*(el_b[i + 1] - el_b[i]) )
        # function fun transformed to the reference element [-1, 1]
        fun2 = lambda s: np.polyval(eta[j], s)
        # shape function
        
        fun12 = lambda s: fun1(s)*fun2(s)
        intval, foo = fixed_quad(fun12, -1, 1)
        funbar[index] += (el_b[i + 1] - el_b[i])/2*intval
        # integral
        if j != d:
                index += 1
    
    return Nel, funbar, Nbase


def lag_fun(cvec, eta, el_b, bcs=2):
    '''Given a coefficient vector, returns a function in the space spanned by a Lagrange basis.
    
        Parameters:
            cvec : ndarray
                Coefficient vector.
            eta : list
                List elements are the shape functions in 'poly1d' format.
            el_b : ndarray
                1D array of element interfaces from left to right including the boundaries. 
            bcs : int
                Specifies the boundary conditions. DEFAULT = 2 which stands for Dirichlet.
                1 stands for periodic.
                
        Returns:
            Nel : int
                The number of elements, Nel = len(el_b) - 1. 
            fun : function
                Function defined on [el_b[0], el_b[-1]].            
    '''
    
    Nel = len(el_b) - 1
    # number of elements
    
    m = len(eta)
    # number of shape functions
    
    d = m - 1
    # polynomial degree
    
    Ntot = Nel*m - (Nel - 1)
    Nbase = Ntot - bcs
    # number of basis functions
    
    def fun(x):
        '''Function in a finite dimensional space spanned by Lagrange basis functions,
        created with fembase.lag_fun.
        '''
        
        hist, bin_edges = np.histogram(x, bins=el_b)
        el = np.nonzero(hist)
        el = el[0][0] # extract the numeric value from tuple
        # element in which x is located
        
        funval = 0
        if el == 0:
            index = 0
            for i in range(1, m): 
                funval += ( cvec[index]*np.polyval( eta[i], 2*(x - el_b[el])
                                                    /(el_b[el + 1] - el_b[el]) - 1 ) )
                index += 1
                
        elif el == Nel - 1:
            index = (Nel - 1)*d - 1
            for i in range(d): 
                funval += ( cvec[index]*np.polyval( eta[i], 2*(x - el_b[el])
                                                    /(el_b[el + 1] - el_b[el]) - 1 ) )
                index += 1
                
        else:
            index = el*d - 1
            for i in range(m): 
                funval += ( cvec[index]*np.polyval( eta[i], 2*(x - el_b[el])
                                                    /(el_b[el + 1] - el_b[el]) - 1 ) )
                index += 1
                
        return funval
    
    return Nel, fun