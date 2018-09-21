'''Basic functions for Particle-In-Cell (PIC) programming.
    Written by Florian Holderied
    florian.holderied@tum.de

   Contains:
       borisPush : function
           Pushes the particles' velocities and positions by a time step dt.
       fieldInterpolation : fuction
           Computes the fields at the particle positions.
'''


import numpy as np
import scipy as sc

def borisPush(particles, dt, Bp, Ep, q, m, L, bcs = 2):
    '''Pushes the particles' velocities and positions by a time step dt.
        
    Parameters:
        particles : ndarray
            2D-array (N_p x 7) containing the positions (x,y,z), velocities (vx,vy,vz) and weights (w) of N_p particles.
        dt: float
            The time step.
        Bp: ndarray
            2D-array (N_p x 3) containing the magnetic field at the particle positions. 
        Ep: ndarray
            2D-array (N_p x 3) containing the electric field at the particle positions.
        q : float
            The electric charge of the particles.
        m : float
            The mass of the particles.
        L : float
            The length of the computational domain.
        bcs : int
            The boundary conditions. DEFAULT = 2 (periodic), 1 (reflecting).

    Returns:
        xnew : ndarray
            2D-array (N_p x 3) with the updated particle positions.
        vnew : ndarray
            2D-array (N_p x 3) with the updated particle velocities.
    '''
    
    if bcs == 2:
        
        qprime = dt*q/(2*m)
        H = qprime*Bp
        S = 2*H/(1 + np.linalg.norm(H, axis = 1)**2)[:,None]
        u = particles[:, 3:6] + qprime*Ep
        uprime = u + np.cross(u + np.cross(u, H), S)
        vnew = uprime + qprime*Ep
        xnew = (particles[:, 0:3] + dt*vnew)%L

        return xnew,vnew
    
    elif bcs == 1:
        
        print('Not yet implemented!')
        
        return 
    
    
    
def fieldInterpolation(xk, el_b, shapefun, ex, ey, ez, bx, by, bz, bcs = 1):
    '''Computes the fields at the particle positions from a FEM Lagrange basis.
    
    Parameters:
        xk : ndarray
            1D-array containing the x-positions of all particles.
        el_b : ndarray
            The element boundaries.
        shapefun: LagrangeShape object
            List with the Lagrange shape functions.
        ex: ndarray
            The coefficients of the x-component of the electric field.
        ey: ndarray
            The coefficients of the y-component of the electric field.
        ez: ndarray
            The coefficients of the z-component of the electric field.
        bx : ndarray
            The coefficients of the x-component of magnetic field.
        by : ndarray
            The coefficients of the y-component of magnetic field.
        bz : ndarray
            The coefficients of the z-component of magnetic field.
        bcs : int
            The boundary conditions. DEFAULT = 1 (Dirichlet), 2 (periodic).
            
    Returns:
        Ep : ndarray
            2D-array (N_p x 3) containing the electric field at the Np particle positions.
        Bp : ndarray
            2D-array (N_p x 3) containing the magnetic field at the Np particle positions.
    '''
    
    if bcs == 1:
        
        N_el = len(el_b) - 1
        N_p = len(xk)
        d = shapefun.d
        
        Ep = np.zeros((N_p, 3))
        Bp = np.zeros((N_p, 3))
        
        exj = np.array([0] + list(ex) + [0])
        eyj = np.array([0] + list(ey) + [0])
        ezj = np.array([0] + list(ez) + [0])
        bxj = np.array([0] + list(bx) + [0])
        byj = np.array([0] + list(by) + [0])
        bzj = np.array([0] + list(bz) + [0])
        
        Xbin = np.digitize(xk, el_b) - 1
        
        for ie in range(0, N_el):
            
            indices = np.where(Xbin == ie)[0]
            s = 2*(xk[indices] - el_b[ie])/(el_b[ie + 1] - el_b[ie])
            
            for il in range(0, d + 1):
                
                i = d*ie + il
                bi = np.polyval(shapefun.eta[il],s)
                
                Ep[indices, 0] += exj[i]*bi
                Ep[indices, 1] += eyj[i]*bi
                Ep[indices, 2] += ezj[i]*bi
                Bp[indices, 0] += bxj[i]*bi
                Bp[indices, 1] += byj[i]*bi
                Bp[indices, 2] += bzj[i]*bi
                
        return Ep,Bp
    
    elif bcs == 2:
        
        N_el = len(el_b) - 1
        N_p = len(xk)
        d = shapefun.d
        Nbase = N_el*d
        
        Ep = np.zeros((N_p, 3))
        Bp = np.zeros((N_p, 3))
        
        Xbin = np.digitize(xk, el_b) - 1
        
        for ie in range(0, N_el):
            
            indices == np.where(Xbin == ie)[0]
            s = 2*(xk[indices] - el_b[ie])/(el_b[ie + 1] - el_b[ie])
            
            for il in range(0, d + 1):
                
                i = d*ie + il
                bi = np.polyval(shapefun.eta[il],s)
                
                Ep[indices, 0] += exj[i%Nbase]*bi
                Ep[indices, 1] += eyj[i%Nbase]*bi
                Ep[indices, 2] += ezj[i%Nbase]*bi
                Bp[indices, 0] += bxj[i%Nbase]*bi
                Bp[indices, 1] += byj[i%Nbase]*bi
                Bp[indices, 2] += bzj[i%Nbase]*bi
                
        return Ep,Bp


    
def assemb_S(x_p, kernel, kernel_supp, el_b, s, siz = 0):
    '''Assembles the S-matrix with elements S_ip = S(x_i - x_p), where S is a smoothing kernel.

    Parameters:
        x_p : ndarray
            1D-array with the particle positions.
        kernel : function
            The smoothing kernel. Symmetric, positive and normalized to 1.
        kernel_supp : float
            The size of the kernel support.
        el_b : ndarray
            The element boundaries including the domain boundaries.
        s : ndarray
            The knot vector on the reference element [-1,1].
        siz : int
            Switch for V1-projection: DEFAULT = 0 (no projection), 2 means S is prepared with 2 additional lines for the V1-projection.
            
    Returns:
        S : sparse matrix
            The matrix with entries S_ip.
        x_vec : ndarray
            The global knot vector.
    '''
    
    
    Nel = len(el_b) - 1
    # number of elements
    
    Np = len(x_p)
    # number of particles
    
    d = len(s) - 1
    # degree of basis functions
            
    Nknots = Nel*d + 1 + siz
    x_vec = np.zeros(Nknots)
    # global knot vector
    
    if siz == 0:
        
        for ie in range(Nel):
            for il in range(d + 1):

                i = ie*d + il
                x_vec[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
                # assemble global knot vector

    elif siz == 2:
        
        xvec[0] = el_b[0] - kernel_supp/2
        xvec[-1] = el_b[-1] + kernel_supp/2
        
        for ie in range(Nel):
            for il in range(d + 1):

                i = ie*d + il + 1
                x_vec[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
                # assemble global knot vector
                
                
    col = np.array([])
    row = np.array([])
    data = np.array([])
    # initialize global col, row and data

    for i in range(Nknots):

        col_i = np.where(np.abs(x_p - x_vec[i]) < kernel_supp/2)[0]
        row_i = np.ones(len(col_i))*i
        data_i = kernel(x_p[col_i] - x_vec[i])

        col = np.append(col, col_i)
        row = np.append(row, row_i)
        data = np.append(data, data_i)



    S = sc.sparse.csr_matrix((data, (row, col)), shape = (Nknots, Np))

    return S, x_vec
    
    
    
    
def computeDensity(particles, q, el_b, kernel, s):
    '''Parameters:
        particles : ndarray
            2D-array (Np x 4) containing the particle information (x,vx,vy,w)
        q : float
            The charge of the particles
        el_b : ndarray
            1D-array specifying the element boundaries
        kernel : function
            The smoothing kernel
        s : ndarray
            The Lagrange interpolation points on the reference element [-1,1]
            
        Returns:
            rho : ndarray
                The coefficients of the charge density
    '''
    
    Nel = len(el_b) - 1
    # number of elements
    
    Np = len(particles[:, 0])
    # number of particles
    
    d = len(s) - 1
    # degree of basis functions
    
    p = d + 1
    # degree of Gauss-Legendre quadrature
    
    Nknots = Nel*d + 1
    glob_s = np.zeros(Nknots)
    # global knot vector
    
    rho = np.zeros(Nel*d)
    # initialize charge density
    
    for ie in range(Nel):
        for il in range(d + 1):
            
            i = ie*d + il
            glob_s[i] = el_b[ie] + (s[il] + 1)/2*(el_b[ie + 1] - el_b[ie])
            # assemble global knot vector
    
    xi,wi = np.polynomial.legendre.leggauss(p)
    # weights and quadrature points on reference element [-1,1]
    
    quad_points = np.zeros(p*(Nknots - 1))
    weights = np.zeros(p*(Nknots - 1))
    # global quadrature points and weights

    for i in range(Nknots - 1):
        a1 = glob_s[i]
        a2 = glob_s[i+1]
        xis = (a2 - a1)/2*xi + (a1 + a2)/2
        quad_points[p*i:p*i + p] = xis
        wis = (a2 - a1)/2*wi
        weights[p*i:p*i + p] = wis
        # assemble global quad_points and weights
                       
    bins = np.digitize(particles[:, 0], glob_s) - 1
    # particle binning in global knot vector
    
    dx = el_b[1] - el_b[0]
    
    for i in range(Nel*d):
        for j in range(Np):
            
            fun = lambda x: kernel(particles[j,0] - x)
            rho[i] += 2*q/dx*particles[j,4]*sc.integrate.quad(fun,glob_s[i],glob_s[i+1])
            
    return rho
        
        