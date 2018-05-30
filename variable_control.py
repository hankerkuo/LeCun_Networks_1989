import numpy as np
from scipy import ndimage


# used to calculate sobel and prewitt operator's threshold
def Relu_threshold(input, threshold=3):
    # cool Relu method!
    output = input * (input >= threshold)
    return output


# input is a 3D-np.array [samples, height, width] , axis = 0v1v2 each refers to y, x and sqrt(y**2 + x**2)
# threshold is like a Relu activation function (e.g. default: threshold = 3, all value will be 0 if <=3)
def sobel_operator_nd(input, threshold=0, axis=2):
    input_y = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    input_x = np.ndarray(shape=(np.shape(input)[1:]), dtype=np.float32)
    output = np.ndarray(shape=(np.shape(input)), dtype=np.float32)
    for _ in range(len(input)):
        for channel in range(np.shape(input)[3]):
            input_y[:, :, channel] = ndimage.sobel(input[_, :, :, channel], 0)
            input_x[:, :, channel] = ndimage.sobel(input[_, :, :, channel], 1)
            if axis == 0:
                output[_, :, :, channel] = input_y[:, :, channel]
            elif axis == 1:
                output[_, :, :, channel] = input_x[:, :, channel]
            elif axis == 2:
                output[_, :, :, channel] = np.sqrt(np.square(input_x[:, :, channel]) + np.square(input_y[:, :, channel]))
                output[_, :, :, channel] = Relu_threshold(output[_, :, :, channel], threshold=threshold)
    return output