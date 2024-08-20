import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2

def solve():
  d = 3
  U, M1, M2 = initialise_input(3, d)
  mask = np.array([[1, 0]])
  mask = np.tile(mask, d//2)
  if d % 2 != 0:
    mask = np.append(mask, [[1]], axis=1)
  inv_mask = 1 - mask
  Z = M1 * mask * mask.T + M2 * inv_mask * inv_mask.T
  max_indices = Z
  print(max_indices)
  '''
  Enter your code here for steps 1 to 6
  '''
  return max_indices
  
solve()
