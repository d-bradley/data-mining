import math
import numpy as np
import matplotlib.pyplot as plt

node_id = [1,2,3,4,5,6]

L = [[2,-1,-1,0,0,0],
     [-1,2,-1,0,0,0],
     [-1,-1,3,-1,0,0],
     [0,0,-1,21,-10,-10],
     [0,0,0,-10,110,-100],
     [0,0,0,-10,-100,110]]

Ls = [[1,(-1/math.sqrt(4)),(-1/math.sqrt(10)),0,0,0],
     [(-1/math.sqrt(4)),1,(-1/math.sqrt(6)),0,0,0],
     [(-1/math.sqrt(10)),-1/math.sqrt(6),1,(-1/math.sqrt(63)),0,0],
     [0,0,(-1/math.sqrt(63)),1,(-10/math.sqrt(2310)),(-10/math.sqrt(2310))],
     [0,0,0,(-10/math.sqrt(2310)),1,(-100/math.sqrt(12100))],
     [0,0,0,(-10/math.sqrt(2310)),(-100/math.sqrt(12100)),1]]

w, u = np.linalg.eig(L)
ws, us = np.linalg.eig(Ls)

print("u:")
print(u)
print()
print("us:")
print(us)

plt.figure(1)
plt.xlabel('Node ID')
plt.ylabel('Eigenvector for Laplacian (u)')
plt.title('Node ID vs. Eigenvector for Laplacian (u)')
plt.plot(node_id, u,'o', color='black')
plt.figure(2)
plt.xlabel('Node ID')
plt.ylabel('Eigenvector for Laplacian Symmetric (us)')
plt.title('Node ID vs. Eigenvector for Laplacian Symmetric (us)')
plt.plot(node_id, us, 'o', color='black')
plt.tight_layout()
plt.show()