from time import clock
import numpy as np
import numpy.random as rn
import numpy.linalg as la
import matplotlib.pylab as mp

def p(txt, x, d=0):
    print(txt+np.array2string(x.astype(np.float) if d else
    x.astype(np.int), formatter={'float_kind':('{0:.' + str(d) + 'f}' if d else '{0:d}').format}))

# <nest> Tablice NumPy<!>cwicz--A.2<!>

print("\nPolecenie 1.1: \n")
a = np.array([5*k for k in range(111)])
b = np.array([i for i in range(111) if i % 5 == 0])
print(a, "\n\n", b)

print("\nPolecenie 1.2: \n")
x = np.zeros(21)
x[::2] = 1

p('x = ', x)

print("\nPolecenie 1.3: \n")
y = np.zeros(30)
y[2:30:3] = 1
p('y = ', y)

print("\nPolecenie 1.4: \n")
X = np.ones((7, 9))
X[::2, ::2] = 0
X[1::2, 1::2] = 0
p('X = ', X)

print("\nPolecenie 1.5: \n")
Y = np.kron(X, np.ones((9, 7)))
np.set_printoptions(threshold=np.nan)
p('Y = ', Y)
numrows = len(Y)
numcols = len(Y[0])
print(numrows, numcols)
mp.imshow(Y)
mp.show()

print("\nPolecenie 1.6: \n")

#for n in [3**4, 3**5, 3**6, 3**7, 3**8]:
for n in [3**8, 3**7, 3**6, 3**5, 3**4]:
    tt = []
    for i in range(10):
        x = rn.rand(n, int(n/3))
        beg = clock()
        y = x.T
        end = clock()
        tt.append((end-beg)*1e6)
        s = "macierz {0:d} x {1:d}, "
        s += "czas: {2:.0f}[mikrosekund(y)]"
        print(s.format(n, int(n/3), min(tt)))

print("\nPolecenie 1.7: \n")
QQ = np.array([[0,1,2], [3,4,5], [6,7,9]])
QQt = QQ.T
print()
print(QQ.strides, "-----", QQt.strides)

print("\nPolecenie 1.8: \n")
X = np.array(np.arange(1, 37))
T = np.resize(X, (3, 3, 4))
p('T= ', T)

# T5 rzedu 5 o wymiarach 2x2x2x2x2, zawierajacy kolejn liczby calkowite [-16,16)

g = np.array(np.arange(-16, 16))
g = np.resize(g, (2, 2, 2, 2, 2))
print('g = ', g)

print("\nPolecenie 1.9: \n")
I = np.where(T % 9 == 0)
print('I =', [r.tolist() for r in I])
p('T[I]= ', T[I])
Ia = np.array(I).T.copy()
p('Indeksy wierszowo I.T:\n', Ia)
print('T[indeksy] =', [T[r[0], r[1], r[2]] for r in Ia])

print("\nPolecenie 1.10: \n")
def toGray(g):
    m, n = g.shape
    G = np.zeros((m, n, 3), dtype=np.uint8)
    G[:, :, 0] = g
    G[:, :, 1] = g
    G[:, :, 2] = g
    return G

RGB = mp.imread('motyl.jpg')
print('Show RGB: ... in Matplotlib')
mp.imshow(RGB)
mp.show()
print('Show R: ... in Matplotlib')
mp.imshow(toGray(RGB[:, :, 0]))
mp.show()
print('Show RGB as vertical flip: ...')
mp.imshow(np.fliplr(RGB))
mp.show()
print('Show RGB rotated by +90 degrees: ...')
mp.imshow(np.rot90(RGB))
mp.show()
print('Show RGB rotated by -90 degrees: ...')
mp.imshow(np.rot90(RGB, k=3))
mp.show()
print('Show G: ... in Matplotlib')
mp.imshow(np.flipud(RGB))
mp.imshow(toGray(RGB[:, :, 0]))
mp.show()

luma = RGB[:, :, 0].astype(np.uint16)

luma += RGB[:, :, 1]
luma += RGB[:, :, 2]
LUMA = (luma/3).astype(np.uint8)
print('Show (R+G+B)/3: ... in Matplotlib')
mp.imshow(np.flipud(np.fliplr(LUMA)))
mp.show()

print("\nPolecenie 1.11: \n")
M = np.min(LUMA)
print('Minimalna jasność motyla:', M)
k = np.where(LUMA==M)[0].size
s = '...osiagana w {0:d} punktach'
print(s.format(k))

print("\nPolecenie 2.1: \n")
A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
At = A.T
print("------------------------------------")
print(A.shape)
print(np.dot(A.shape, A.strides))
print(np.dot(A.T.shape, A.T.strides))
for i in range(4):
    for j in range(4):
        print(np.dot(np.array([i, j]), A.strides), "<><>", np.dot(np.array([j, i]), At.strides))
print("------------------------------------")


p('A=\n', A)

print("\nPolecenie 2.2: \n")
U, s, Vt = la.svd(A)
ns = s.size

p('wartosci sygularne, 16 cyfr:\n', s, d=16)
eps = 10**(-15)
small = np.where(s<eps)
neps = len(small)
r = ns if neps==0 else small[0][0]
print('rzad macierzy :', r)

print("\nPolecenie 2.3: \n")

print("\nPolecenie 2.4: \n")
B_S = U[:, :r]
p('baza dla S=spann[A], B_S=\n', B_S, d=3)

print("\nPolecenie 2.5: \n")
print(np.dot(B_S.T, B_S))

print("\nPolecenie 2.6: \n")
V=Vt.T
B_K=V[:,r:]
print('wymiar kernela :', A.shape[1]-r)
p('baza kernela :\n', B_K, d=3)