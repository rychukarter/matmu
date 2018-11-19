# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy.linalg as la
import matplotlib.pylab as mp
from numpy.lib.stride_tricks import as_strided
import acwicz as ac

# Strumień funkcji dla zadań 16-10-2017
def Kochański():
    about_pi = math.sqrt(40/3.0-2*math.sqrt(3))
    s = 'Aproksymacja pi według Kochańskiego: '
    print((s+'{:.15f}').format(about_pi))
    s = 'Błąd aproksymacji: '
    print((s+'{:.15f}').format(math.pi-about_pi))
def DrabinkaOporników(eps):
    R = r = 1 
    phi = (math.sqrt(5)+1)/2 # złota liczba
    n = 0
    while (phi-R/r)>eps:
        n += 1
        R = r+ (R*r)/(r+R)
    s = 'Drabinka oporników o n = {:d} oczkach '
    s += 'ma opór zastępczy równy {:.15f}'
    print(s.format(n,R))
    s = 'Opór ten jest równy oporowi nieskończonej'
    s += '  drabinki z błędem epsilon = {:.5e}'
    print(s.format(eps))
def PodziałFibonacciego(eps):
    F = Fp = 1; n =2
    phi = (math.sqrt(5)+1)/2 # złota liczba
    while math.fabs(phi-F/Fp)>eps:
        n += 1
        Fp,F = F,Fp+F
    s = 'Kolejne wyrazy ciągu Fibonacciego '
    s += 'stanowią złotą proporcję {:.15f}\n'
    s += 'z błędem epsilon = {:.5e} '
    s += 'poczynając od n = {:d}'
    print(s.format(F/Fp,eps,n))    
def toGray(g):
    m,n = g.shape
    G = np.zeros((m,n,3),dtype=np.uint8)
    G[:,:,0] = g; G[:,:,1] = g; G[:,:,2] = g
    return G

def toGrayVirtual(g):
    yres, xres= g.shape
    s0, s1 = g.strides
    G = as_strided(g,shape=(yres, xres, 3), strides=(s0,s1,0))
    return G

def toLuma(RGB):
    luma = RGB[:,:,0].astype(np.uint16)
    luma += RGB[:,:,1]
    luma += RGB[:,:,2]
    return (luma/3).astype(np.uint8)
def hat(u):
    return np.array([[0,-u[2],u[1]],
                     [u[2],0,-u[0]],
                     [-u[1],u[0],0]])
def dehat(uhat):
    u = np.array([uhat[2,1],uhat[0,2],uhat[1,0]])
    return u


def Rodrigez(u,phi):
    uhat = hat(u)
    cphi = math.cos(phi)
    R = cphi*np.eye(3)+(1-cphi)*np.outer(u,u)+\
        math.sin(phi)*uhat
    return R


def deRodrigez(R):
    cphi = (np.trace(R)-1)/2
    phi = math.acos(cphi)
    uhat = (R-R.T)/(2*math.sin(phi))
    u = dehat(uhat)
    return u,phi


def quaternionFromAxisAngle(u,phi):
    imag = math.sin(phi/2)*u
    return np.array([math.cos(phi/2),
                     imag[0],imag[1],imag[2]])


def rotationMatrixFromQuaternion(q):
    q0p = q[0]*q[0]-0.5
    R = 2*np.array([
      [q0p+q[1]*q[1],q[1]*q[2]-q[0]*q[3],
                   q[1]*q[3]+q[0]*q[2]],
      [q[1]*q[2]+q[0]*q[3],q0p+q[2]*q[2],
                         q[2]*q[3]-q[0]*q[1]],
      [q[1]*q[3]-q[0]*q[2],q[2]*q[3]+q[0]*q[1],
                         q0p+q[3]*q[3]]
      ])
    return R
def slerp(q0,q1,tau):
    cos01 = np.dot(q0,q1)
    omega = math.acos(cos01)
    sino = math.sin(omega)
    qtau = (math.sin((1-tau)*omega)*q0+
            math.sin(tau*omega)*q1)/sino
    return qtau
if __name__=='__main__':
    # Strumień wywołań funkcji dla zadań 16-10-2017
    Kochański()
    DrabinkaOporników(1e-10)
    PodziałFibonacciego(1e-10)
    RGB = mp.imread('motyl.jpg')
    print('Show RGB: ... in Matplotlib')
    mp.imshow(RGB)
    mp.show()
    print('Show R: ... in Matplotlib')
    mp.imshow(toGray(RGB[:,:,0]))
    mp.show()
    print('Show RGB as vertical flip: ...')
    mp.imshow(np.fliplr(RGB));
    mp.show()
    print('Show RGB rotated by +90 degrees...')
    mp.imshow(np.rot90(RGB))
    mp.show()
    print('Show RGB rotated by -90 degrees...')
    mp.imshow(np.rot90(RGB,k=3))
    mp.show()
    
    print('Show (R+G+B)/3: ... in Matplotlib')
    LUMA = toLuma(RGB)
    mp.imshow(toGray(LUMA)); mp.show()
    mp.imshow(toGrayVirtual(LUMA)); mp.show()
    M = np.max(LUMA)
    print('Maksymalna jasność motyla: ',M)
    k = np.where(LUMA==M)[0].size
    s = '...osiągana w {0:d} punktach'
    print(s.format(k)) 
    a = np.ones(3); b = np.array([3,5,7.])
    print('Norma wektora a: {:.3f}'.\
           format(la.norm(a)))
    print('...pierwiastek z 3: {:.3f}'.\
           format(math.sqrt(3)))
    print('Norma wektora b: {:.3f}'.format(la.norm(b)))
    print('Iloczyn skalarny wektorów a i b: {:.3f}'.\
          format(np.dot(a,b)))
    print('Iloczyn zewnętrzny wektorów a i b:\n',
          np.outer(a,b))
    print('Iloczyn zewnętrzny wektorów b i a:\n',
          np.outer(b,a))
    print('Iloczyn wektorowy a i b:\n',np.cross(a,b))
    
    R = Rodrigez([0,0,1],math.pi/4)
    print('R =\n',R)
    u,phi = deRodrigez(R)
    print('u =',u,'phi =',phi*180/math.pi)
    q = quaternionFromAxisAngle(u,phi)
    print('kwaternion q =',q)
    R = rotationMatrixFromQuaternion(q)
    print('R =\n',R)
    q0 = np.array([1,0,0,0.])
    q1 = quaternionFromAxisAngle(u,math.pi/2)
    q2 = slerp(q0,q1,0.5)
    R2 = rotationMatrixFromQuaternion(q2)
    print('R2 =\n',R2)