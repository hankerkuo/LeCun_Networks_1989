import numpy as np
from scipy import ndimage


# used to calculate sobel and prewitt operator's threshold
def Relu_threshold(input, threshold=3):
    for height in range(np.shape(input)[0]):
        for width in range(np.shape(input)[1]):
            if input[height, width] <= threshold:
                input[height, width] = 0
    output = input
    return output


# input is a 3D-np.array [samples, height, width] , axis = 0v1v2 each refers to y, x and sqrt(y**2 + x**2)
# threshold is like a Relu activation function (e.g. default: threshold = 3, all value will be 0 if <=3)
def sobel_operator(input, threshold=3, axis=2):
    input_y = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    input_x = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    output = np.ndarray(shape=(np.shape(input)), dtype=np.float32)
    for _ in range(len(input)):
        input_y[:, :] = ndimage.sobel(input[_, :, :], 0)
        input_x[:, :] = ndimage.sobel(input[_, :, :], 1)
        if axis == 0:
            output[_, :, :] = input_y[:, :]
        elif axis == 1:
            output[_, :, :] = input_x[:, :]
        elif axis == 2:
            output[_, :, :] = np.sqrt(np.square(input_x[:, :]) + np.square(input_y[:, :]))
            output[_, :, :] = Relu_threshold(output[_, :, :], threshold=threshold)
    return output


def prewitt_operator(input, threshold=3, axis=2):
    input_y = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    input_x = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    output = np.ndarray(shape=(np.shape(input)), dtype=np.float32)
    for _ in range(len(input)):
        input_y[:, :] = ndimage.prewitt(input[_, :, :], 0)
        input_x[:, :] = ndimage.prewitt(input[_, :, :], 1)
        if axis == 0:
            output[_, :, :] = input_y[:, :]
        elif axis == 1:
            output[_, :, :] = input_x[:, :]
        elif axis == 2:
            output[_, :, :] = np.sqrt(np.square(input_x[:, :]) + np.square(input_y[:, :]))
            output[_, :, :] = Relu_threshold(output[_, :, :], threshold=threshold)
    return output


def laplacian_operator(input):
    output = np.ndarray(shape=(np.shape(input)), dtype=np.float32)
    for _ in range(len(input)):
        output[_, :, :] = ndimage.laplace(input[_, :, :])
    return output


def gaussian_laplace_operator(input):
    output = np.ndarray(shape=(np.shape(input)), dtype=np.float32)
    for _ in range(len(input)):
        output[_, :, :] = ndimage.gaussian_laplace(input[_, :, :], sigma=1)
    return output



