
# DenseNet implementation and training with TensorFlow 2.x


## Requirements

+ Internet connection for downloading CIFAR10 dataset and TensorFlow docker image.
+ Docker and Nvidia runtime installed as described on [TensorFlow website](https://www.tensorflow.org/install/docker)


## Scripts in this repository

+ densenet.py: Description of the DenseNet architecture
+ train.py: Training and evaluation of the densenet model using the well-known CIFAR-10 dataset.
+ run_train.py: Linux shell script that sets the arguments for train.py and executes it.
  + The python command can be executed directly if preferred, for example:
  
```shell
  python train.py --opt rms --epochs 140 --learnrate 0.001 --batchsize 125 --tboard ./tb_log --keras_hdf5 ./densenet.h5
```
  
+ start_tf2.sh: Linux shell script that pulls latest TensorFlow nightly build and starts docker.

## Instructions 

1. Clone or download/unzip this repository to a folder. 
2. Navigate into the folder created in Step 1.
3. Open a command shell/terminal and start the TensorFlow docker by running start_tf2.sh:

```shell
source ./start_tf2.sh
```

4. When the docker starts, execute run_train.sh like this:

```shell
source ./run_train.sh
```


## References

1. Huang et al. <a href="https://arxiv.org/pdf/1608.06993.pdf">"Densely Connected Convolutional Networks" (v5) Jan 28 2018</a>.
2. Krizhevsky, Alex. <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">"Learning Multiple Layers of Features from Tiny Images"</a>.

