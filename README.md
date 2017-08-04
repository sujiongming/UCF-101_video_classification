# Video classification with CNN(InceptionV3)

Classify UCF-101 videos using one frame at a time with a CNN(InceptionV3)

The paper blog [**视频分类之UCF101上的CNN方法详解**] for chinese at zhihu.com [https://zhuanlan.zhihu.com/p/28307781](https://zhuanlan.zhihu.com/p/28307781) 

## Requirements
This project runs at ubuntu 16.04 with GeForce GTX 1080 8G X2.

This code requires you have Keras 2+ and TensorFlow 1+ or greater installed. 

## Getting the data

First, download the dataset from UCF into the `data` folder:

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Then extract it with `unrar e UCF101.rar`.

Next, create folders (still in the data folder) with `mkdir train && mkdir test && mkdir sequences && mkdir checkpoints`.

Now you can run the scripts in the data folder to move the videos to the appropriate place, extract their frames and make the CSV file the rest of the code references. You need to run these in order. Example:

`python 1_move_files.py`

`python 2_extract_files.py`  # make sure installed 'ffmpeg' before,eg. sudo apt-get install ffmpeg 

## Running models

Run `python CNN_train_UCF101.py` to train and save the CNN model.

Choose the best model to run `python CNN_evaluate_testset.py` to evaluate the whole test set, that takes a long time for 697,865 images.

Run `python CNN_validate_images.py` to classify a few images.

### UCF101 Citation

Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild., CRCV-TR-12-01, November, 2012. 

This project thanks for the open project at github [https://github.com/harvitronix/five-video-classification-methods](https://github.com/harvitronix/five-video-classification-methods "Five video classification methods")
