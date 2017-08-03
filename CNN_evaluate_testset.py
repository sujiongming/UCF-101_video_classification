"""
Classify test images set through our CNN.
Use keras 2+ and tensorflow 1+
It takes a long time for hours.
"""
import numpy as np
import operator
import random
import glob
from UCFdata import DataSet
from processor import process_image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

data = DataSet()
def main(nb_images=5):
    # CNN model evaluate

    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    test_data_num = 697865 #the number of test images
    batch_size = 32
    test_generator = test_data_gen.flow_from_directory('./data/test/', target_size=(299, 299),
                                                       batch_size=batch_size, classes=data.classes,
                                                       class_mode='categorical')
    # load the trained model that has been saved in CNN_train_UCF101.py, your model name maybe is not the same as follow
    model = load_model('data/checkpoints/inception.057-1.16.hdf5')
    results = model.evaluate_generator(generator=test_generator, steps=test_data_num // batch_size)
    print(results)
    print(model.metrics)


if __name__ == '__main__':
    main()
