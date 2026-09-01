import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

p_nr = np.load('dev/cache_nnraw_A.npy')
p_n1 = np.load('dev/cache_nn_n1_A.npy')
print(f'nn_raw vs N1 예측상관 = {np.corrcoef(p_nr, p_n1)[0,1]:.4f}')
