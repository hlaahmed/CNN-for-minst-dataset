# CNN-for-minst-dataset
Model layers and hyperparameters: 
1. Convolutional layer:
        1. 32 filters, each of size 3x3
        2. Activation: relu 
        3. initialize the kernel filter weights with 'he_uniform'
        4. we know that the size of our images is 3x28x28, so we enter it as the input_shape  
        5. the stride in the convolution layer is set by default to 1 6. Padding is valid padding 

2. Max pooling layer: 
        1. pool size is 2x2 
        2.  we didn't specify the stride, so the default is that it will be the same as pool size 
        3. Padding is valid padding 
3. Flatten layer 
4. Dense layer: 1. the output size of the layer is 100 2. The activation chosen is relu 3. ‘He_unifrom’ initialization 
5. Dense layer: 1. the output size of the layer is 10 2. The activation chosen is softmax 
 
The chosen gradient descent is stochastic gradient descent, the chosen learning parameter is 0.01 and the chosen momentum is 0.9 The chosen batch size is 128 The chosen number of iterations is 20 
