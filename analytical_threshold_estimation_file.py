import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math
import sys
pi = math.pi
from scipy.special import gammainc , gamma

#Numerical skeleton of the analytical estimation for the threshold of PBH formation in terms of the linear 
#component of the compaction function C_l, expressed in terms of κ (the curvature of C_l at the peak value r_m). 
#See the paper on Arxiv for more details.The skeleton provided is a simple example
#following the equation Eq.(16) in the paper. The example is very basic, with the curvature fluctuations 
#already expressed in terms of C_l​ and κ, and is shown here for simplicity and for provide understanding
#to the reader. The user can modify it for more complex shape parametrizations at her/his convinience


#Analytical formula from ArXiv:1907.13311, through the parameter 'q'
def delta_q(qvalue):
    delta_c_q = (4./15.)*np.exp(-1./qvalue)*(qvalue**(1.-5./(2.*qvalue)))/(gamma(5./(2*qvalue))*(gammainc(5./(2*qvalue),1./qvalue)) )	
    return delta_c_q

#we define a cubic spline to relate κ in terms of delta_q
def interpolant():
    lista_q = np.logspace(np.log10(0.015),np.log10(1000),10**4)
    kappa_list = []
    Cl_list = []
    for k in range(len(lista_q)):
        qq = lista_q[k]
        deltac = delta_q(lista_q[k])

        #threshold_lista.append(deltac)
        kappa = qq*4.0*deltac*np.sqrt(1.-(3./2.)*deltac)
        kappa_list.append(kappa)
        Cl_I = (4./3.)*(1.-np.sqrt(1.-3.*deltac/2.)) #corresponds to the type-I region
        Cl_list.append(Cl_I)
    return kappa_list , Cl_list

### we make the cubic spline interpolation 
kappa_list , Cl_list = interpolant()
spline_kappa_deltacl = CubicSpline(kappa_list , Cl_list)
###

# new analytical estimation with the parametter κ
def delta_analytical_estimation(kappa):
    kappa_cross = 21.327 #value from the paper
    if kappa<kappa_cross:
        delta_l_c = spline_kappa_deltacl(kappa)
    else:
        aa = 0.519620930
        bb = 0.266869266
        delta_l_c = aa*(kappa**bb)
    return delta_l_c

#Curvature profile. We take the one of Eq.(16) in Arxiv
def zeta(rr,cl,kappa,rmf):
    tilde_kappa = kappa/cl
    #shape of zeta
    zeta_shape =(3.*math.e/4.)*(cl/(np.sqrt(tilde_kappa)))\
    *np.exp(-(rr/rmf)**(np.sqrt(tilde_kappa)))
    return zeta_shape

#Compaction function C and linear component C_l
def compaction_functions(rr,zeta_der):
    #shape of zeta
    Cl_shape = -(4./3.)*rr*zeta_der
    C_shape = Cl_shape-(3./8)*Cl_shape**2
    return Cl_shape,C_shape

#Computation of the kappa value for the given profile
def kappa_computation(rr,rmax,Cl_shape_input):
    spline = CubicSpline(rr,Cl_shape_input)
    der_2_Cl = spline.derivative(2) #we compute the second derivative of Cl
    kappa_value = -(rmax**2)*der_2_Cl(rmax) 
    return kappa_value

rmf=1.0 #location of r_m from the profile
xx = np.linspace(0.,2*rmf,10**3) #array for the radial coordinate
kappa_m = 10. #value of kappa that we fix for the profile

Cl_min = (4./3.)*(1. - np.sqrt(2./5.))  #minimum value of Cl
Cl_max = 4.0  #maximum value that we set for searching the threshold
array_Cl_values = np.linspace(Cl_min,Cl_max,10**3) #array for value of Cl for iteration
resol = 0.001 #resolution to set to find the threshold with the analytical estimate

thres_array = []
kappa_array = []

for Cl_item in array_Cl_values: #we iterate

    zeta_profile = zeta(xx,Cl_item,kappa_m,rmf)
    spline_zeta = CubicSpline(xx,zeta_profile)
    zeta_profile_der  = spline_zeta.derivative(1)(xx) 
    Cl_profile , C_profile= compaction_functions(xx,zeta_profile_der) #compute the compaction functions
    kk = kappa_computation(xx,rmf,Cl_profile)
    #critical value of the linear compaction function at r_m
    delta_l_c = delta_analytical_estimation(kk)
    thres_array.append(delta_l_c)

    #we stop the iteration once the threshold is within the desired resolution
    if abs(Cl_item-delta_l_c)<resol:
        print("Critical value of Cl_c , κ and δl_c:", Cl_item,kk,delta_l_c)
        break
