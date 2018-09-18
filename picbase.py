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

def borisPush(particles, dt, Bp, Ep, q, m, L, bcs == 2):
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
            
            indices == np.where(Xbin == ie)[0]
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
        