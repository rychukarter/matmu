import numpy as np
import numpy.linalg as la
import matplotlib.pylab as mp

def p(txt,x,d=0):
    print(txt+np.array2string(
    x.astype(np.float) if d
    else x.astype(np.int),
    formatter = {
        'float_kind':('{0:.'+str(d)+'f}' if d
                        else '{0:d}').format
    }))

a = np.array([(i+1) for i in range(100) if ((i+1)%3 == 0)])
print(a)

b = np.ones(20)
b[::2]=0
print('x = ', b)

c = np.zeros(100)
c[1::5]=1
print('c = ', c)

d = np.array([[j+i for i in range(8)] for j in range(8)])
print('d = ', d)

e = np.ones([8,8])
e[::2,::2] = 0
e[1::2,1::2] = 0
print('e = ', e)

f = np.kron(e, np.ones((4,4)))
print('f = ', f)

mp.imshow(f)
mp.show()

g = np.array(np.arange(-16, 16                    ))
g = np.resize(g,(2,2,2,2,2))
print('g = ', g)

A = np.array([[9,6,3], [8,5,2], [7,4,1]])
print('A = ', A)
U, s, Vt = la.svd(A)
ns = s.size
eps = 10**(-15)
small = np.where(s<eps)
neps = len(small)
r = ns if neps==0 else small[0][0]

print('Rzad macierzy: ', r)

B_S = U[:,:r]
p('baza dla spann[A], B_S=\n', B_S, d=3)

print('sprawdzenie ortogonalności:\n', np.dot(B_S.T, B_S))
V = Vt.T
B_K = V[:, r:]
print('wymiary kernela:', A.shape[1]-r)
p('baza kernela:\n', B_K, d=3)

n = np.cross(U[:,0], U[:,1])
nn = [-0.408, 0.816, -0.408]
alpha = np.arccos(np.dot(n,nn))
print(180 - alpha*180/np.pi)

v=V[:,2]
P_A = np.eye(3)-np.outer(v,v)/np.dot(v,v)
p('macierz rzutu na spann[A]:\n', P_A, d=3)

q = np.dot(P_A, [1,0,0])
print(np.dot(q,n))
H_A = np.eye(3) - 2*np.outer(v,v)/np.dot(v,v)


