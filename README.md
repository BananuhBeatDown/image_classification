# Classifying Images using a Neural Network

This program classifies images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) using a Convolutional Neural Network (CNN). The images are preprocessed and normalized then used to train a CNN consisting of convolutional, max pooling, dropout, fully connected, and output layers.


In-depth analysis and examples can be found in [dlnd_image_classification.ipynb](https://github.com/BananuhBeatDown/image_recognition/blob/master/dlnd_image_classification.ipynb).

## Install

- Python 3.6
    + I recommend installing [Anaconda](https://www.continuum.io/downloads) as it is alreay set up for machine learning
    + If unfamiliar with the command line there are graphical installs for macOS, Windows, and Linux
- [TensorFlow](https://www.tensorflow.org/install/?nav=true)
- [tqdm](https://github.com/noamraph/tqdm)

## Dataset

The dataset is broken into batches to prevent your machine from running out of memory.  The CIFAR-10 dataset consists of 5 batches, named `data_batch_1`, `data_batch_2`, etc.. Each batch contains the labels and images that are one of the following:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Parameters

#### image_recognition.py

You can view different example of the CIFAR-10 dataset by changing the values of the `batch_id` and the `sample_id`:

- `batch_id` - id for a batch (1-5)
- `sample_id` - id for a image and label pair in the batch

```python
batch_id = 1
sample_id = 5
helper.display_stats(cifar10_dataset_folder_path, batch_id, sample_id)
```

<img src="https://user-images.githubusercontent.com/10539813/27656181-52f23142-5c48-11e7-8f39-7a204c6d11eb.png" width="256">

#### train_image_recognition.py

You can experiment with the CNN by altering:
- `depth` - Alter the depths of the CNN layers using common memory sizes
    + 64
    + 128
    + 256
    + ...
- `epochs` - number of training iterations
- `batch_size` - set to highest number your machine has memory for using common memory sizes
- `keep_probability` - probability of keeping node using dropout

## Example Output

**Command Line**   

`python image_recognition.py`   

**You must press [enter] to continue after example image appears.*  

<img src="https://user-images.githubusercontent.com/10539813/27656180-52effbb6-5c48-11e7-92ae-3a8793db00a6.png" width="512">

`python train_image_recognition.py`  

<img src="https://user-images.githubusercontent.com/10539813/27656695-3ef257a6-5c4a-11e7-8644-eb4df95054f4.png" width="512">

## License
The image_classification program is a public domain work, dedicated using [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). I encourage you to use it, and enhance your understanding of CNNs and the deep learning concepts therein. :)
