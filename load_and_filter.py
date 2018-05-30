from six.moves import cPickle as pickle
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np

# extract data
with open('./train_data/data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open('./train_data/label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open('./test_data/data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open('./test_data/label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

# building ndarrays for storing results after filters
tr_dat_after_sobel = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
tr_dat_after_prewitt = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
tr_dat_after_laplacian = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
tr_dat_after_gaussian_laplace = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)

te_dat_after_sobel = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
te_dat_after_prewitt = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
te_dat_after_laplacian = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
te_dat_after_gaussian_laplace = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)

# filter operations on training data
for _ in range(320):
    tr_dat_after_sobel[_, :, :] = ndimage.sobel(tr_dat[_, :, :], 0)
    tr_dat_after_prewitt[_, :, :] = ndimage.prewitt(tr_dat[_, :, :], 0)
    tr_dat_after_laplacian[_, :, :] = ndimage.laplace(tr_dat[_, :, :])
    tr_dat_after_gaussian_laplace[_, :, :] = ndimage.gaussian_laplace(tr_dat[_, :, :], sigma=1)

# filter operations on test data
for _ in range(160):
    te_dat_after_sobel[_, :, :] = ndimage.sobel(te_dat[_, :, :], 0)
    te_dat_after_prewitt[_, :, :] = ndimage.prewitt(te_dat[_, :, :], 0)
    te_dat_after_laplacian[_, :, :] = ndimage.laplace(te_dat[_, :, :])
    te_dat_after_gaussian_laplace[_, :, :] = ndimage.gaussian_laplace(te_dat[_, :, :], sigma=1)


print(tr_dat[2])
print(tr_dat_after_prewitt[2])
plt.imshow(tr_dat[1], cmap='gray')
plt.show()
