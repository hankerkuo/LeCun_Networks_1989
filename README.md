# LeCun_Networks_1989
Implementation of simple CNN architecture proposed by Yann LeCun in 1989


This project is implemented in Tensorflow and Keras.
We use pickle files as our input format (our input is 16 by 16 handwritten digit data), it is possible to use different kinds of input format by editing the code a little bit.

Since the pickle files are in the folder : test_data and train_data, if you download all of the repo including folders test_data and train_data, all you need to do is to run net1.py ~ net2.py.

We also did some work in reprocessing the data by several filter operators:
Sobel, Prewitt, Laplacian and Gaussian Laplace

After preprocessing the data with operators , we trained and tested them in the net5, which is the finest architecture in this work. The results were visualized by using tensorboard and available in our Final report.

Most of the experimental results are shown in our paper, which is also available here in this repo.
