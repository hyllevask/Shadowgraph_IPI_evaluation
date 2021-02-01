import numpy as np
def fringes2size(N,m,lamb,f_num,theta):
    
    alfa = 2*np.arcsin(1/f_num/2)

    A = 2*lamb/alfa
    B = (np.cos(theta/2) + (m*np.sin(theta/2)) / np.sqrt(m**2-2*m*np.cos(theta/2) + 1))

    d = N*A/B
    return d
d = fringes2size(1,1.33,0.532,4,76/90*np.pi/2)
print(d)