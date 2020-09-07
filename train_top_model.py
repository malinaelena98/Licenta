import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import keras
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import Sequential

# bottleneck files
top_model_weights_path = "bottleneck_features_model.h5"

# where to save the model
save_model = "trained_model_bottleneck_features.h5"

# images format
# all the images are resized  at 224x224(resize_imgs.py)
img_width = 224
img_height = 224

# paths to training,validation and test folders
training_set = 'dataset\\train'
validation_set = 'dataset\\validation'
test_set = 'dataset\\test'
batch_size = 32

# augment data from training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# for test set and validation set, just rescaling
validation_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)


def train_top_model():
    # training data
    generator_top_train = train_datagen.flow_from_directory(
        training_set,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False)

    num_classes = len(generator_top_train.class_indices)

    # load the bottleneck features which was created earlier in extract_bottleneck_features
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top_train.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    # validation data
    generator_top_valid = validation_datagen.flow_from_directory(
        validation_set,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False)

    # load the bottleneck features which was created earlier in  extract_bottleneck_features
    validation_data = np.load('bottleneck_features_validation.npy')

    # get the class lebels for the validation data, in the original order
    validation_labels = generator_top_valid.classes

    # convert the validation labels to categorical vectors
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    # testing data
    generator_top_test = test_datagen.flow_from_directory(
        test_set,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False)

    # load the bottleneck features which was created earlier in bottleneck_features()
    test_data = np.load('bottleneck_features_test.npy')

    # get the class lebels for the validation data, in the original order
    test_labels = generator_top_test.classes

    # convert the testing labels to categorical vectors
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    # On top of the bottleneck features  we are going to train a small model which will have target labels specific to our dataset
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))

    # adding Dropout to avoid overfitting
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    # adding Dropout to avoid overfitting
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    checkpointer = ModelCheckpoint(filepath='bottleneck_features.h5', monitor='val_acc', verbose=1, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_acc', verbose=1, patience=5)
    history = model.fit(train_data, train_labels,
                        epochs=50,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[checkpointer, early_stopping])

    # save the model's weight values which were learned during training
    model.save_weights(top_model_weights_path)

    #
    # (eval_loss_train, eval_accuracy_train) = model.evaluate(train_data, train_labels, batch_size=batch_size, verbose=1)
    # print("Accuracy: {:.2f}%".format(eval_accuracy_train * 100))
    # print("Loss: {}".format(eval_loss_train))
    #
    # (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)
    # print("Accuracy: {:.2f}%".format(eval_accuracy * 100))
    # print("Loss: {}".format(eval_loss))
    #
    # (eval_loss_test, eval_accuracy_test) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=1)
    # print("Accuracy: {:.2f}%".format(eval_accuracy_test * 100))
    # print("Loss: {}".format(eval_loss_test))


if __name__ == "__main__":
    train_top_model()
