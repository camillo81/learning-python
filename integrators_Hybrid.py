import numpy as np
import scipy as sc


def integrator_HE(ex, ey, bx, by, yx, yy, vx, vy, G, Q0, eps0, wpe, qe, me, dt):
    
    bx_new = bx + dt*np.dot(G, ey)
    by_new = by - dt*np.dot(G, ex)
    
    yx_new = yx + dt*eps0*wpe**2*ex
    yy_new = yy + dt*eps0*wpe**2*ey
    
    vx_new = vx + dt*qe/(eps0*me)*Q0.transpose().dot(ex)
    vy_new = vy + dt*qe/(eps0*me)*Q0.transpose().dot(ey)
    
    return bx_new, by_new, yx_new, yy_new, vx_new, vy_new


def integrator_HB(ex, ey, bx, by, mass_0_inv, G, mass_1, c, dt):
    
    mat = np.dot(mass_0_inv, np.dot(np.transpose(G), mass_1)) 
    ex_new = ex + dt*c**2*np.dot(mat, by)
    ey_new = ey - dt*c**2*np.dot(mat, bx)
    
    return ex_new, ey_new


def integrator_HY(ex, ey, yx, yy, eps0, wce, dt):
    
    
    ex_new = ex - 1/(eps0*wce)*(yx*np.sin(wce*dt) - yy*np.cos(wce*dt) + yy)
    ey_new = ey - 1/(eps0*wce)*(yy*np.sin(wce*dt) + yx*np.cos(wce*dt) - yx)
    
    yx_new = yx*np.cos(wce*dt) + yy*np.sin(wce*dt)
    yy_new = yy*np.cos(wce*dt) - yx*np.sin(wce*dt)
    
    return ex_new, ey_new, yx_new, yy_new


def integrator_Hx(ex, vx, vz, Q0, W, By, mass0_inv, eps0, qe, me, dt):
    
    ex_new = ex - dt*qe/eps0*np.dot(mass_0_inv, Q0.dot(W.dot(vx)))
    
    vz_new = vz + dt*qe/me*By.dot(vx)
    
    return ex_new, vz_new


def integrator_Hy(ey, vy, vz, Q0, W, Bx, mass0_inv, eps0, qe, me, dt):
    
    ey_new = ey - dt*qe/eps0*np.dot(mass_0_inv, Q0.dot(W.dot(vy)))
    
    vz_new = vz - dt*qe/me*Bx.dot(vy)
    
    return ey_new, vz_new



def integrator_Hz(bx, by, z, vx, vy, vz, el_b, Lz, shapefun, qe, me, dt):
    
    Np = len(z)
    Nel = len(el_b) - 1
    p = len(bx)/Nel
    
    z_new = (z + dt*vz)%Lz
    
    
    
    row_IQ = np.array([])
    col_IQ = np.array([])
    dat_IQ = np.array([])
    
    
    t_vec = np.zeros((Nel, Np))
    t_all = Lz/np.abs(vz)
    
    for iy in range(Nel):
        t_vec[iy] = ((el_b[iy] - z)/vz)%t_all
        
    
    
    ind_t_min = np.argmin(t_vec, axis = 0)
    signs = np.sign(vz).astype(int)
    steps = np.heaviside(vz, 1).astype(int)
    part_num  = np.arange(Np)
    parts = np.full(Np, True, dtype = bool)
    
    
    t_lower = np.zeros(Np)
    
    pos_lower = z
    iy = 0
    
    
    while np.any(parts) == True:
            
        ind_now = ind_t_min + signs*iy
        t_now = t_vec[ind_now%Nel, part_num]
        element = (ind_now - steps)%Nel
        
        bol = t_now > dt
        
        pos_upper = el_b[element + steps]
        pos_upper[bol] = z_new[bol]


        for il in range(p):
            int_vals = np.zeros(sum(parts))
            
            for m in range(il + 1, p + 1):
                int_vals += 1/vz[parts]*(el_b[element[parts] + 1] - el_b[element[parts]])/2*(shapefun.eta[m](2*(pos_upper[parts] - el_b[element[parts]])/(el_b[element[parts] + 1] - el_b[element[parts]]) - 1) - shapefun.eta[m](2*(pos_lower[parts] - el_b[element[parts]])/(el_b[element[parts] + 1] - el_b[element[parts]]) - 1)) 

            
            row_IQ = np.append(row_IQ, element[parts]*p + il)
            col_IQ = np.append(col_IQ, part_num[parts])
            dat_IQ = np.append(dat_IQ, int_vals)

        pos_lower = el_b[element + steps]
        
        pos_lower[np.logical_and(pos_lower == el_b[0], signs == -1)] = el_b[-1]
        pos_lower[np.logical_and(pos_lower == el_b[-1], signs == 1)] = el_b[0]
        
        
        parts[bol] = False
        iy += 1
        
        
    
    
    
    
    IQ = sc.sparse.csr_matrix((dat_IQ, (row_IQ, col_IQ)), shape = (Nel*p, Np)) 
    
    Bx_vec = IQ.transpose().dot(bx)
    By_vec = IQ.transpose().dot(by)
    
    IBx = sc.sparse.csr_matrix((Bx_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    IBy = sc.sparse.csr_matrix((By_vec, (np.arange(Np), np.arange(Np))), shape = (Np, Np))
    
    
    vx_new = vx - qe/me*IBy.dot(vz)
    vy_new = vy + qe/me*IBx.dot(vz)
    
    return z_new, vx_new, vy_new
