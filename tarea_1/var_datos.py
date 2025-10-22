import numpy as np
import matplotlib.pyplot as plt

N = 1000
N_nds = 10
p = 3
X = np.zeros( (N, N_nds) )
A = np.zeros( (N_nds, N_nds, p) )

std_mattrans = 0.2
for j in range( p ):
    A[0, 3, j] = std_mattrans * np.random.randn()
    A[1, 0, j] = std_mattrans * np.random.randn()
    A[2, 1, j] = std_mattrans * np.random.randn()
    A[3, 2, j] = std_mattrans * np.random.randn()

std_datos = 0.1
for t in range( p, N ):
    X[t, :] = std_datos * np.random.randn( N_nds )
    for j in range( p ):
        X[t, :] = X[t, :] + A[:,:,j] @ X[t-j-1, :]

fig, axs = plt.subplots(nrows = N_nds, ncols = 1)
for nd in range(N_nds):
    axs[ nd ].plot( X[:, nd] )
plt.show()

A_ = np.sum( A, axis=2 )
A_[ A_!=0 ] = 1
plt.imshow(A_)
plt.show()