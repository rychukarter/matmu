import numpy as np
from numpy.linalg import norm
from math import sqrt
from time import clock
import matplotlib.pylab as mp
from acwicz import acwicz as ac

def hat(u):
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

a = np.array([1,2,3])
b = np.array([-1,-1,1])
ones3 = np.ones(3)
adot = np.dot(a, ones3)
aout = np.outer(np.ones(5), a)
print(adot, aout)

A=np.outer(a, np.ones(5))
B=np.outer(np.ones(5), a)
print(B==A.T)

I = np.array(range(10000))
print(I.shape)
print(I[:20])
C=np.outer(I,np.ones(10000))

start=clock()
D=C.T
end=clock()
print(end-start)

A=np.zeros((1000,1000))
start=clock()
D=A.T
end=clock()
print(end-start)

B=np.zeros((1000,10000))
start=clock()
E=B.T
end=clock()
print(end-start)

print(A.strides, D.strides)

print(np.dot(a,b))
c=np.cross(a,b)
print(c)

print(np.dot(b,c))
P=norm(a)*norm(b)
print(P)
print(norm(c))

def toLuma(RGB):
    luma = RGB[:,:,0].astype(np.uint16)
    luma += RGB[:,:,1]
    luma += RGB[:,:,2]
    return (luma/3).astype(np.uint8)

RGB = mp.imread('motyl.jpg')
print(RGB.shape, RGB.size/3)
RGBlr = np.fliplr(RGB)
RGB90 = np.rot90(RGB)
RGB270 = np.rot90(RGB, k=3)
RGBlum = toLuma(RGB)
mp.imshow(RGBlum)
mp.show()
from numpy.random import randn

R=ac.Rodrigez([0,0,1], np.pi/4)
print(R)

cloud = randn(100)
cloud3 = np.zeros((3, 100))
cloud3[0, :] = cloud
mp.plot(cloud3[0,:], cloud3[1,:], 'go')
cloud3rot = np.dot(R, cloud3)
mp.plot(cloud3[0,:], cloud3[1,:], 'ro')
#mp.show()

u=[0,0,1]
phi=np.pi/4
up,phip = ac.deRodrigez(R)
print(up, phip)

q0=np.array([1,0,0,0])
q1=np.array([np.cos(np.pi/8),0,0,np.sin(np.pi/8)])
q2=ac.slerp(q0,q1,0.5)
R2 = ac.rotationMatrixFromQuaternion(q2)
print(R2)
u3, phi3 = ac.deRodrigez(R2)
print(u3, phi3, phi3*180/np.pi)
mp.plot(cloud3rot[0,:],cloud3rot[1,:], 'yo')

cloud3slerp = np.dot(R2, cloud3)
mp.plot(cloud3slerp[0,:],cloud3slerp[1,:], 'go')
mp.show()

