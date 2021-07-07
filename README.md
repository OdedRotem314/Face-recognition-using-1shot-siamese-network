# Face-recognition-using-1shot-siamese-network
use CNN siamese network to carry out the task of facial recognition based on the paper Siamese Neural Networks for One-shot Image Recognition
A one-shot learning task for previously unseen objects. Given two facial images of previously unseen persons, the
architecture successfully determine whether they are the same person.
dataset - Labeled Faces in the Wild

Dataset analysis:
Train size – 2145 samples of 250x250 pixels equally divided to 2 classes (same person pairs, different person pairs)
Validation size – the train set above was split so a small validation set can be used (2.5%, 55 samples)
Test size – 1000 samples of 250x250 pixels equally divided to 2 classes (same person pairs, different person pairs)
Preprocessing:
The images were normalized 0-1 GL (divided by 255) and precision set to float32
The data was shuffled so random pairs would be included in each batch
A tensor of length 5 was given to the model (None, 2, 250, 250, 1)
The labels array is a simple 1D vector of 0 or 1 depending on the relevant pair class
Model setup:
Each pair of images is passed through a CNN model, at first feature maps are constructed by convolutional layers. These are later flattened and a dense layer per image is used to calculate a distance metric. This is then finally passed to a 1 neuron layer so a loss can be calculated together with the relevant label (0,1).   
Layers: Main repeating block includes CONV2DDropout
For CONV2D we tried different number of filters (32,64,128) of different sizes (10,7,5,3)
We used stride instead of max-pooling as in the article. Stride=2
We used 'relu' activation s in the article and a l2 regularizer on the weights
Dropout percentage was adjusted to optimize overfit (0.3,0.5,0.7)  0.5
After flattening a dense layer of size 1024 was used (a larger size led to OOM issues)
Distance metric: We used the simple L1 distance as described in the article (other Euclidean measurements are also suitable but were not experimented with)  


Final model architecture:
conv2D(64 filters, 3x3, stride=2, relu, l2 regularizer) 
dropout(0.5)
conv2D(64 filters, 3x3, stride=2, relu, l2 regularizer) 
dropout(0.5)
flatten
dense(1024, sigmoid)
l1 distance calc
dense(1, sigmoid)

Model compiler:
 Loss – ' binary_crossentropy' loss was used due to the problem at hand
Optimizer – 'Adam' and 'RMSprop' where experimented with, with learning rates e-3 to e-6

Model fit:
Batch size – 32, 64 and 128 batch sizes were compared  64 revealed best results 
Stop criteria – we ran 30 epochs for each experiment (different hyper parameters) and chose the best epoch from each experiment. Best meaning highest accuracy on test set.  
Reasoning for hyperparameters and architecture:
We started off implementing the article's architecture
 
This architecture resulted in OOM issues as well as poor results and overfitting.
We reduced the model layer size to 2 convolution blocks and reduced the number of filters as well as the dense layer number of neurons from 4096 to 1024.
Hyper parameters optimization – we optimized each HP sequentially (one at a time) by selected 3-5 working points for each. During training the epoch with the highest accuracy on validation was considered to be the best for that working point.   
Hyper-parameters working points and performances:
Initial conditions:  4 conv layers of 64 filters of size (3,3), stride(2,2), drop out(0.5) followed by dense(64). Batch_size = 32 , Adam(0.0006)
Number of layers [2,3,4,5]  2 - test accuracy of 0.68
Number of filters [32,64,128]  64  - test accuracy of 0.68
Filter size [3,5,7]  3 - test accuracy of 0.7
Dense nodes [64,512,1024]  1024 - test accuracy of 0.7
Optimizer [adam, rmsprop]  adam - test accuracy of 0.71
Batch_size [32,64,128]  64 - test accuracy of 0.75
Lr [0.006, 0.0006, 0.00006, 0.000006]  0.00006 - test accuracy of 0.7
final conditions:  2 conv layers of 64 filters of size (3,3), stride(2,2), drop out(0.5) followed by dense(64). Batch_size = 64 , Adam(0.00006)  accuracy 0.75

Performance:
Convergence times – time for a single epoch=6sec
Loss and accuracy of the best configuration: 
train_loss = 0.36, train_acc =0.99 , val_loss =0.92 , val_acc =0.67, test_acc = 0.65









Graphs of final configuration:
 

Examples of correct/incorrect classification

Why is the model insufficient?
1. due to OOM issues we cannot build a deep enough network to include more weights needed to approximate the data distribution. 
2.  The small training dataset  (2145) limits the model training as well.
3. The large image size of 250x250 is another issue. Usually neural networks work better with smaller images. This leads to OOM and GPU limitations. 
3. The prediction could be improved for this type of problem by introducing triplet loss which learns the difference between a similar pair and different pair of the same original image. 
  
