from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import cv2
from tkinter import *
import tkinter.font as font
from PIL import ImageTk, Image
from tkinter import filedialog

img_width = 224
img_height = 224
batch_size = 32

top_model_weights_path = "bottleneck_features_model.h5"

classes = ['benign', 'malign']


def predict(image_path):
    image = load_img(image_path, target_size=(img_height, img_width))
    image = img_to_array(image)
    # rescale
    image = image / 255
    image = np.expand_dims(image, axis=0)
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))

    # adding Dropout to avoid overfitting
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))

    # adding Dropout to avoid overfitting
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    return class_predicted[0]


def select_image():
    global panelA, panelB

    # open a file chooser to select input
    path = filedialog.askopenfilename()

    # if file path was selected
    if len(path) > 0:
        preds = predict(path)
        class_pred = classes[preds]

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format
        image = Image.fromarray(image)
        image = image.resize((224, 224), Image.ANTIALIAS)

        # convert to to ImageTk format
        image = ImageTk.PhotoImage(image)

    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        # the first panel will store our original image
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="top", padx=10, pady=10)

        # while the second panel will store the prediction
        myFont = font.Font(family='Helvetica', size=16, weight='bold')
        panelB = Label(text=class_pred, height=4, width=10, bg='#E4CC7A')
        panelB['font'] = myFont
        panelB.text = class_pred
        panelB.pack(side="bottom", padx=5, pady=5)

    # otherwise, update the image panels
    else:
        # update the pannels
        panelA.configure(image=image)
        panelB.configure(text=class_pred)
        panelA.image = image
        panelB.text = class_pred


root = Tk()
root.wm_title("Detect skin cancer")
root.geometry("600x600")
root.configure(background='#E4CC7A')

panelA = None
panelB = None
myFont = font.Font(family='Helvetica', size=16, weight='bold')

# create a button, then when pressed, will trigger a file chooser

btn = Button(root, text="Select an image", height=4, width=15, command=select_image, bg="#EEE1A3")
btn['font'] = myFont
btn.pack(side="bottom")

if __name__ == "__main__":
    root.mainloop()
