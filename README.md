# Custom Batch Normalization

With this repo I present my custom implementation of [batch normalization](https://arxiv.org/pdf/1502.03167.pdf).

### Main Motivation

The original paper motivates the need of batch normalization 
by claiming that the training process of a neural network is 
complicated by the fact that the inputs to each layer are 
affected by the parameters of all preceding layers. Therefore, 
small changes to the network parameters amplify as the network 
becomes deeper, and increase the internal covariant shift.
They define *internal covariate shift* as the change in the
distribution of network activations due to the change in
network parameters during training. This phenomenon harms the
training process in both final accuracy and required time. By
fixing the distribution of the layer inputs as the training
progresses, the training process is expected to improve. As a 
matter of fact, it has been long known that the network training 
converges faster if its inputs are *whitened* (linearly transformed 
to have zero means and unit variances). Doing so to input images
is an important step in neural network training, with this paper
the authors introduced a novel method to yield *withened* inputs to 
every single layer.

### Practical Steps
First of all, batch normalization is performed slightly differently 
during training and at inference time.
#### During the training process, for each batch:
1. The mean is computed.
   
2. The variance is computed.
   
3. The computed mean and variance are used to update dataset statistics by
means of a moving average. These statistics will be employed during inference.

4. Tensors are normalized by subtracting the batch mean and dividing 
   for the batch variance (to which a small epsilon value is summed).
   
5. Given the fact that normalizing each input of a layer may change 
   what the layer can represent, tensors are further scaled by multiplying 
   them for learned parameter *gamma*, and shifted by summing learned
   parameter *beta* to them. This step ensures that each transformation 
   inserted in the network can represent the identity transform.
   
#### During inference, for each image:

1. tensors are normalized, by subtracting the **dataset mean** and dividing 
   for the **dataset variance** (to which a small epsilon value is summed)
   
2. Step 5 of the training process is performed, although obiouvsly parameters
gamma and beta will not be updated.


The employment of dataset statistics during inference ensures *determinism*:
the network output only depends on the single corresponding input.

### Using the Code

In order to use the provided code:

```
python main.py 
```
Will launch the training of LeNet **without** batch normalization.

```
python main.py --bn
```
Will launch the training of LeNet **with** batch normalization.

The official pytorch MNIST download link seems to be unavailable, the dataset can be manually 
downloaded by using

```
wget www.di.ens.fr/~lelarge/MNIST.tar.gz
tar -zxvf MNIST.tar.gz
```
### Results

All experiments were run on a single K80, and evaluated by means of the balanced accuracy metric.

|  BN | Learning Rate | Final Acc | Epochs to reach  98.5%   | Average training  time(s) for epoch |
|:---:|---------------|:---------:|--------------------------|-------------------------------------|
| Yes | 0.07          |    99.2   |            13            |                 10.5                |
| No  | 0.07          |    98.9   |            33            |                 10.8                |
| Yes | 0.3           |    99.2   |             7            |                 10.5                |
| No  | 0.3           |    96.7   |           Never          |                 10.8                |


By adding batch normalization after every layer but the last one (both convolutional
and fully-connected), the final accuracy on the test set is boosted from `98.9%`to `99.2%`.
Moreover, batch normalization makes the accuracy raise much faster, thus speeding 
up the training process.
Adding the proposed custom implementation of batch normalization slightly increases the 
required training time for epoch.
Finally, by using the `--learning_rate 0.3` parameter users can test how using a larger
learning rate harms the training process of the original version of LeNet, 
whereas the network that employs batch normalization slightly improves.


#### Confusion Matrix

```
[[ 976    0    0    0    0    0    2    2    0    0]
 [   0 1131    0    2    0    0    2    0    0    0]
 [   1    0 1025    0    0    0    0    5    1    0]
 [   0    1    1 1002    0    2    0    1    3    0]
 [   0    0    0    0  975    0    2    1    1    3]
 [   2    0    0    4    0  881    4    0    0    1]
 [   2    3    0    1    3    2  946    0    1    0]
 [   0    2    8    0    2    0    0 1015    0    1]
 [   0    1    3    0    1    1    0    1  963    4]
 [   1    0    0    0    6    2    0    5    1  994]]
```