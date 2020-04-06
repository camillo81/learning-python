import numpy as np
import scipy as sc

from psydac.core.interface import collocation_matrix
from numba import jit


@jit
def integrator_HE(ex, ey, bx, by, yx, yy, vx, vy, G, Q0, eps0, wpe, qe, me, dt):
    """
    Solves analytically the ODE system corresponding to Hamiltonian of the electric field.
    
    Parameters
    ----------
    ex : np.array
        old electric field coefficients (x-component)
        
    ey : np.array
        old electric field coefficients (y-component)
        
    bx : np.array
        old magnetic field coefficients (x-component)
        
    by : np.array
        old magnetic field coefficients (y-component)
        
    yx : np.array
        old cold current density coefficients (x-component)
        
    yy : np.array
        old cold current density coefficients (y-component)
        
    vx : np.array
        old particle velocities (x-component)
        
    vy : np.array
        old particle velocities (y-component)
        
    G : np.array
        discrete gradient matrix
    
    Q0 : sparse matrix
        basis functions of V0 evaluated at old particle positions
        
    eps0 : float
        vacuum permittivity
        
    wpe : float
        cold electron plasma frequency
        
    qe : float
        electron charge
        
    me : float
        electron mass
    
    dt : float
        time step
    
    
    Returns
    -------
    bx_new : np.array
        new magnetic field coefficients (x-component)
        
    by_new : np.array
        new magnetic field coefficients (y-component)
        
    yx_new : np.array
        new cold current density coefficients (x-component)
        
    yy_new : np.array
        new cold current density coefficients (y-component)
        
    vx_new : np.array
        new particle velocities (x-component)
        
    vy_new : np.array
        new particle velocities (y-component)
    """
    
    
    bx_new = bx + dt*np.dot(G, ey)
    by_new = by - dt*np.dot(G, ex)
    
    yx_new = yx + dt*eps0*wpe**2*ex
    yy_new = yy + dt*eps0*wpe**2*ey
    
    vx_new = vx + dt*qe/me*Q0.transpose().dot(ex)
    vy_new = vy + dt*qe/me*Q0.transpose().dot(ey)
    
    return bx_new, by_new, yx_new, yy_new, vx_new, vy_new



def integrator_HE_full(ex, ey, ez, bx, by, yx, yy, yz, vx, vy, G, Q0, eps0, wpe, qe, me, dt):
    
    bx_new = bx + dt*np.dot(G, ey)
    by_new = by - dt*np.dot(G, ex)
    
    yx_new = yx + dt*eps0*wpe**2*ex
    yy_new = yy + dt*eps0*wpe**2*ey
    yz_new = yz + dt*eps0*wpe**2*ez
    
    vx_new = vx + dt*qe/me*Q0.transpose().dot(ex)
    vy_new = vy + dt*qe/me*Q0.transpose().dot(ey)
    
    return bx_new, by_new, yx_new, yy_new, yz_new, vx_new, vy_new


def integrator_HB(ex, ey, bx, by, mass_0_inv, G, mass_1, c, dt):
    
    mat = np.dot(mass_0_inv, np.dot(np.transpose(G), mass_1)) 
    ex_new = ex + dt*c**2*np.dot(mat, by)
    ey_new = ey - dt*c**2*np.dot(mat, bx)
    
    return ex_new, ey_new



@jit(nopython=True)
def integrator_HY(ex, ey, yx, yy, eps0, wce, dt):
    
    ex_new = ex - 1/(eps0*wce)*(yx*np.sin(wce*dt) - yy*np.cos(wce*dt) + yy)
    ey_new = ey - 1/(eps0*wce)*(yy*np.sin(wce*dt) + yx*np.cos(wce*dt) - yx)
    
    yx_new = yx*np.cos(wce*dt) + yy*np.sin(wce*dt)
    yy_new = yy*np.cos(wce*dt) - yx*np.sin(wce*dt)
    
    return ex_new, ey_new, yx_new, yy_new


def integrator_HY_full(ex, ey, ez, yx, yy, yz, eps0, wce, dt):
    
    ex_new = ex - 1/(eps0*wce)*(yx*np.sin(wce*dt) - yy*np.cos(wce*dt) + yy)
    ey_new = ey - 1/(eps0*wce)*(yy*np.sin(wce*dt) + yx*np.cos(wce*dt) - yx)
    ez_new = ez - eps0*dt*yz
    
    yx_new = yx*np.cos(wce*dt) + yy*np.sin(wce*dt)
    yy_new = yy*np.cos(wce*dt) - yx*np.sin(wce*dt)
    
    return ex_new, ey_new, ez_new, yx_new, yy_new


@jit
def integrator_Hx(ex, vx, vy, vz, Q0, By, W, mass_0_inv, eps0, qe, me, wce, dt):
    
    ex_new = ex - dt*qe/eps0*np.dot(mass_0_inv, Q0.dot(W.dot(vx)))
    
    vy_new = vy - dt*wce*vx
    
    vz_new = vz + dt*qe/me*By.dot(vx)
    
    return ex_new, vy_new, vz_new


@jit
def integrator_Hy(ey, vx, vy, vz, Q0, Bx, W, mass_0_inv, eps0, qe, me, wce, dt):
    
    ey_new = ey - dt*qe/eps0*np.dot(mass_0_inv, Q0.dot(W.dot(vy)))

    vx_new = vx + dt*wce*vy
    
    vz_new = vz - dt*qe/me*Bx.dot(vy)
    
    return ey_new, vx_new, vz_new





def integrator_Hz(p, bx, by, z, vx, vy, vz, el_b, Lz, qe, me, dt, t):
    
    Np    = len(z)                  # ... number of particles
    Nel   = len(el_b) - 1           # ... number of elements
    dz    = Lz/Nel                  # ... element size
    Nbase = Nel + p                 # ... number of basis functions
    
    z_new = (z + dt*vz)%Lz          # ... new particle positions
    
    
    row_IQ = np.array([])           # ... basis function indices
    col_IQ = np.array([])           # ... particle indices
    dat_IQ = np.array([])           # ... line integral results
    
    
    t_vec = np.zeros((Nel, Np))     # ... times particles need to the next element boundary
    t_all = Lz/np.abs(vz)           # ... times particles need to fly through the whole domain
    
    for iy in range(Nel):
        t_vec[iy] = ((el_b[iy] - z)/vz)%t_all
        
    
    ind_t_min = np.argmin(t_vec, axis = 0)     # ... indices of element boundaries that particles reach first
    signs = np.sign(vz).astype(int)            # ... do particles go left (-1) or right (+1)?
    steps = np.heaviside(vz, 1).astype(int)    # ... is the particle on the left/right side of the element boundary?
    part_num  = np.arange(Np)                  # ... particle indices
    
    
    # ... boolean array of particles that need to be further integrated (True: not finished, False: finished)
    parts = np.full(Np, True, dtype = bool)    
    
    t_lower = np.zeros(Np)
    
    pos_lower = np.copy(z)
    iy = 0
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p - 1)
    
    while np.any(parts) == True:
            
        ind_now = ind_t_min + signs*iy
        t_now = t_vec[ind_now%Nel, part_num]
        element = (ind_now - steps)%Nel
        
        bol = t_now > dt
        
        pos_upper = el_b[element + steps]
        pos_upper[bol] = z_new[bol]
        
        wts = (pos_upper - pos_lower)[parts, None]/2*wts_loc
        pts = (pos_upper - pos_lower)[parts, None]/2*pts_loc + (pos_upper[parts, None] + pos_lower[parts, None])/2
        
        coll = collocation_matrix(p - 1, Nbase - 1, t, pts.flatten())/dz
        coll[:, :(p - 1)] += coll[:, -(p - 1):]
        coll = coll[:, :coll.shape[1] - (p - 1)]
        
        particles_left = parts.sum()

        row_IQ, col_IQ, dat_IQ = kernel_lineintegrals(p, particles_left, element[parts], Nbase, vz[parts], wts, coll, part_num[parts], row_IQ, col_IQ, dat_IQ)

        pos_lower = el_b[element + steps]
        
        pos_lower[np.logical_and(pos_lower == el_b[0], signs == -1)] = el_b[-1]
        pos_lower[np.logical_and(pos_lower == el_b[-1], signs == 1)] = el_b[0]
        
        
        parts[bol] = False
        iy += 1
        
        
    
    IQ = sc.sparse.csr_matrix((dat_IQ, (row_IQ, col_IQ)), shape = (Nbase - 1 - (p - 1), Np)) 
    
    Bx_vec = IQ.transpose().dot(bx)
    By_vec = IQ.transpose().dot(by)
    
    IBx = sc.sparse.csr_matrix((Bx_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    IBy = sc.sparse.csr_matrix((By_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    
    
    vx_new = vx - qe/me*IBy.dot(vz)
    vy_new = vy + qe/me*IBx.dot(vz)
    
    return z_new, vx_new, vy_new




@jit
def kernel_lineintegrals(p, particles_left, element, Nbase, vz, wts, coll, part_num, row_IQ, col_IQ, dat_IQ):
    for il in range(p):
        int_vals = np.zeros(particles_left)
        i_glob = (element + il)%(Nbase - 1 - (p - 1))


        for g in range(p - 1):
            int_vals += 1/vz*wts[:, g]*coll[np.arange(particles_left)*(p - 1) + g, i_glob]

        row_IQ = np.append(row_IQ, i_glob)
        col_IQ = np.append(col_IQ, part_num)
        dat_IQ = np.append(dat_IQ, int_vals)
                
    return row_IQ, col_IQ, dat_IQ
    


    
    
    
def integrator_Hz_full(ez, bx, by, z, vx, vy, vz, mass_0_inv, W, el_b, Lz, qe, me, eps0, dt, T):
    
    Np    = len(z)                  # ... number of particles
    Nel   = len(el_b) - 1           # ... number of elements
    dz    = Lz/Nel                  # ... element size
    Nbase = Nel + p                 # ... number of basis functions
    t = T[1:-1]                     # ... knot vector of V1
    
    z_new = (z + dt*vz)%Lz          # ... new particle positions
    
    
    row_IQ_0 = np.array([])         # ... basis function indices (V0)
    col_IQ_0 = np.array([])         # ... particle indices (V0)
    dat_IQ_0 = np.array([])         # ... line integral results (V0)
    
    row_IQ_1 = np.array([])         # ... basis function indices (V1)
    col_IQ_1 = np.array([])         # ... particle indices (V1)
    dat_IQ_1 = np.array([])         # ... line integral results (V1)
    
    t_vec = np.zeros((Nel, Np))     # ... times particles need to the next element boundary
    t_all = Lz/np.abs(vz)           # ... times particles need to fly through the whole domain
    
    for iy in range(Nel):
        t_vec[iy] = ((el_b[iy] - z)/vz)%t_all
        
    
    ind_t_min = np.argmin(t_vec, axis = 0)     # ... indices of element boundaries that particles reach first
    signs = np.sign(vz).astype(int)            # ... do particles go left (-1) or right (+1)?
    steps = np.heaviside(vz, 1).astype(int)    # ... is the particle on the left/right side of the element boundary?
    part_num  = np.arange(Np)                  # ... particle indices
    
    
    # ... boolean array of particles that need to be further integrated (True: not finished, False: finished)
    parts = np.full(Np, True, dtype = bool)    
    
    t_lower = np.zeros(Np)
    
    pos_lower = np.copy(z)
    iy = 0
    
    pts_loc_0, wts_loc_0 = np.polynomial.legendre.leggauss(p)
    pts_loc_1, wts_loc_1 = np.polynomial.legendre.leggauss(p - 1)
    
    while np.any(parts) == True:
            
        ind_now = ind_t_min + signs*iy
        t_now = t_vec[ind_now%Nel, part_num]
        element = (ind_now - steps)%Nel
        
        bol = t_now > dt
        
        pos_upper = el_b[element + steps]
        pos_upper[bol] = z_new[bol]
        
        wts_0 = (pos_upper - pos_lower)[parts, None]/2*wts_loc_0
        pts_0 = (pos_upper - pos_lower)[parts, None]/2*pts_loc_0 + (pos_upper[parts, None] + pos_lower[parts, None])/2
        
        wts_1 = (pos_upper - pos_lower)[parts, None]/2*wts_loc_1
        pts_1 = (pos_upper - pos_lower)[parts, None]/2*pts_loc_1 + (pos_upper[parts, None] + pos_lower[parts, None])/2
        
        coll_0 = collocation_matrix(p, Nbase, T, pts_0.flatten())
        coll_0[:, :p] += coll_0[:, -p:]
        coll_0 = coll_0[:, :coll_1.shape[1] - p]
        
        coll_1 = collocation_matrix(p - 1, Nbase - 1, t, pts_1.flatten())/dz
        coll_1[:, :(p - 1)] += coll_1[:, -(p - 1):]
        coll_1 = coll_1[:, :coll_1.shape[1] - (p - 1)]
        
        
        for il in range(p + 1):
            int_vals = np.zeros(parts.sum())
            i_glob = (element[parts] + il)%(Nbase - p)


            for g in range(p):
                int_vals += 1/vz[parts]*wts_0[:, g]*coll_0[np.arange(parts.sum())*p + g, i_glob]

            row_IQ_0 = np.append(row_IQ_0, i_glob)
            col_IQ_0 = np.append(col_IQ_0, part_num[parts])
            dat_IQ_0 = np.append(dat_IQ_0, int_vals)
        
        
        for il in range(p):
            int_vals = np.zeros(parts.sum())
            i_glob = (element[parts] + il)%(Nbase - 1 - (p - 1))


            for g in range(p - 1):
                int_vals += 1/vz[parts]*wts_1[:, g]*coll_1[np.arange(parts.sum())*(p - 1) + g, i_glob]

            row_IQ_1 = np.append(row_IQ_1, i_glob)
            col_IQ_1 = np.append(col_IQ_1, part_num[parts])
            dat_IQ_1 = np.append(dat_IQ_1, int_vals)
        

        pos_lower = el_b[element + steps]
        
        pos_lower[np.logical_and(pos_lower == el_b[0], signs == -1)] = el_b[-1]
        pos_lower[np.logical_and(pos_lower == el_b[-1], signs == 1)] = el_b[0]
        
        
        parts[bol] = False
        iy += 1
        
        
    IQ0 = sc.sparse.csr_matrix((dat_IQ_0, (row_IQ_0, col_IQ_0)), shape = (Nbase - p, Np)) 
    IQ1 = sc.sparse.csr_matrix((dat_IQ_1, (row_IQ_1, col_IQ_1)), shape = (Nbase - 1 - (p - 1), Np)) 
    
    Bx_vec = IQ1.transpose().dot(bx)
    By_vec = IQ1.transpose().dot(by)
    
    IBx = sc.sparse.csr_matrix((Bx_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    IBy = sc.sparse.csr_matrix((By_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    
    
    vx_new = vx - qe/me*IBy.dot(vz)
    vy_new = vy + qe/me*IBx.dot(vz)
    
    ez_new = ez - eps0*qe*np.dot(mass_0_inv, IQ0.dot(W.dot(vz)))
    
    return ez_new, z_new, vx_new, vy_new