from keras import applications
import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.client import device_lib

# images format
# all the images are resized  at 224x224(resize_images.py)
img_width = 224
img_height = 224

# Create a bottleneck file
top_model_weights_path = "bottleneck_features_model.h5"
saved_model = "trained_model_bottleneck_features.h5"

# paths to training,validation and test folders
training_set = 'dataset\\train'
validation_set = 'dataset\\validation'
test_set = 'dataset\\test'
batch_size = 32

# need this for numpy bottleneck files
# just rescaling

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def bottleneck_features():
    # Loading vgg16 model
    # here we are using the transfer learning
    vgg16 = applications.VGG16(include_top=False, weights='imagenet')

    # using a generator to read images from all the subdirectories from dataset\\train
    # we will generate batches of images
    train_generator = train_datagen.flow_from_directory(training_set,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode="binary",
                                                        shuffle=True)
    # total number of images for training
    count_training_img = len(train_generator.filenames)

    # number of classes
    num_clasees = len(train_generator.class_indices)

    # useful for compute the number of steps
    predict_size_train = int(math.ceil(count_training_img / batch_size))

    # bottleneck features for training
    bottleneck_features_train = vgg16.predict_generator(train_generator, steps=predict_size_train)

    # save the output as a numpy array
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # using a generator to read images from all the subdirectories from dataset\\validation
    # we will generate batches of images
    validation_generator = validation_datagen.flow_from_directory(validation_set,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode="binary",
                                                                  shuffle=True)

    # total number of images for validation
    count_validation_img = len(validation_generator.filenames)

    # useful for compute the number of steps
    predict_size_validation = int(math.ceil(count_validation_img / batch_size))

    # bottleneck features for validation
    bottleneck_features_validation = vgg16.predict_generator(validation_generator, steps=predict_size_validation)

    # save the output as a numpy array
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

    # using a generator to read images from all the subdirectories from dataset\\test
    # we will generate batches of images
    test_generator = test_datagen.flow_from_directory(
        test_set,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True)

    # total number of images for testing
    count_test_samples = len(test_generator.filenames)

    # useful for compute the number of steps
    predict_size_test = int(math.ceil(count_test_samples / batch_size))

    # bottleneck features for test
    bottleneck_features_test = vgg16.predict_generator(test_generator, predict_size_test)

    # save the output as a numpy array
    np.save('bottleneck_features_test.npy', bottleneck_features_test)


if __name__ == "__main__":
    bottleneck_features()
