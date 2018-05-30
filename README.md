# LeCun_Networks_1989 in Tensorflow
## Implementation of simple CNN architecture proposed by Yann LeCun in 1989

- This project is implemented in Tensorflow and Keras.

- We use pickle files as our input format (our input is 16 by 16 handwritten digit data), it is possible to use different kinds of input format by editing the code a little bit.

- Since the pickle files are in the folder : [test_data] and [train_data], if you download all of the repo including folders [test_data] and [train_data], all you need to do is to run [net1.py] ~ [net5.py].

## We also did some works in preprocessing the data by several filter operators:

- Sobel 
- Prewitt
- Laplacian
- Gaussian Laplace
    
  After preprocessing the data with these operators , we trained and tested them in the [net5], which is the finest architecture in this work. The results were visualized by using tensorboard and available in our [paper].

## Results of our experiment

- Most of the experimental results are shown in our paper, which is also available [here] in this repo.

- To get more clear concept of our implementation, we highly recommand to read the [original paper] from Yann LeCun

[here]:<https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/Convolutional%20Neural%20Network%20for%20Handwritten%20Digit%20Recognition.pdf>
[net1.py]:https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/net1.py
[net5.py]:https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/net5.py
[net5]:https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/net5.py
[test_data]:https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/test_data
[train_data]:https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/train_data
[original paper]:http://yann.lecun.com/exdb/publis/pdf/lecun-89.pdf
[paper]:<https://github.com/hankerkuo/LeCun_Networks_1989/blob/master/Convolutional%20Neural%20Network%20for%20Handwritten%20Digit%20Recognition.pdf>
