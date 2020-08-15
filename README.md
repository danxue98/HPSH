# HPSH - Deep Hash Remote Sensing Image Retrieval with Hard Probability Sampling

Firstly, we would like to thank for the authours of ** DEEP METRIC AND HASH-CODE LEARNING FOR CONTENT-BASED RETRIEVAL OF REMOTE SENSING IMAGES** 
accepted in the International Conference on Geoscience and Remote Sensing Symposium (**IGARSS**) 
held in *Valencia, Spain* in July, 2018. Our coding work is based on their code.


# Prerequisites
- Python 2.7
- Tensorflow GPU 1.2.0
- Scipy 1.1.0
- Pillow 5.1.0

**N.B.** The code has only been tested with Python 2.7 and Tensorflow GPU 1.2.0. Higher versions of the software should also work properly.

In our paper we have experimented with two remote sensing benchmark archives - [**UC Merced Data Set**](http://weegee.vision.ucmerced.edu/datasets/landuse.html) (UCMD) and [**AID**](https://arxiv.org/abs/1608.05167).  

# Usage
First, download [UCMD](http://weegee.vision.ucmerced.edu/datasets/landuse.html) dataset or the [AID](https://captain-whu.github.io/AID/) dataset and save them on the disk. For the UCMD dataset the parent folder will contain 21 sub-folders, each containing 100 images for each category. Whereas, for the AID dataset the parent folder will contain 30 sub-folders, cumulatively containing 10,000 images.

N.B.: Code for the respective data sets have been provided in separate folders.

Next, download the pre-trained Tensorflow models following the instructions in [this](https://www.tensorflow.org/tutorials/image_recognition) page.

To extract the feature representations from a pre-trained model:  <br>
  `$ python extract_features.py \`  
    `--model_dir=your/localpath/to/models \`  
    `--images_dir=your/localpath/to/images/parentfolder \`  
    `--dump_dir=dump_dir/`  

To prepare the training and the testing set: <br>
  `$ python dataset_generator.py --train_test_split=0.6`
  
To train the network. It is to be noted that same `train_test_split` should be used as above: <br>
  `$ python trainer.py\`  
  `--HASH_BITS=32 --ALPHA=0.2 --BATCH_SIZE=90\`  
  `--ITERS=10000 --train_test_split=0.6`

To evaluate the performance and save the retrieved samples:<br>
  `$ python eval.py --k=20 --interval=10`
