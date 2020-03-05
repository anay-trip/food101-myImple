# set matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from configurations import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def plot_training(H, N, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)

# derive the path to training, validation and test directory
trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.TEST])

# determine the total number of image paths in training directory
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))
# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
	preprocessing_function = vgg16.preprocess_input
)

# initialize the validation/test data augmentation object
# (adding mean subtraction to this)
valAug = ImageDataGenerator(
	preprocessing_function = vgg16.preprocess_input
)

# initialize the training generators
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	classes = config.CLASSES,
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE)

#initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	classes = config.CLASSES,
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

'''
# initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	classes = config.CLASSES,
	preprocessing_fuction = vgg16.preprocess_input,
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)
'''
# load the VGG16 network, ensuring the head fully connected layer is chopped off
baseModel = vgg16.VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# placing the headmodel on top of the base model whose head is chopped off
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all the layers of base model and freeze
for layer in baseModel.layers:
	layer.trainable = False

# compile model after freezing layers
opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network for a few epochs.
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=50
)
model.save(config.MODEL_PATH)


'''
# reset testing generator and evaluate the network after
# fine tuning just the network head
print("***evaluating after fine-tuning network head...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
plot_training(H, 50, config.WARMUP_PLOT_PATH)
'''
# reset the data generator
trainGen.reset()
valGen.reset()

# now that the head FC layers have been trained/initialized,
# unfreezing some of the final set of CONv layers
for layer in baseModel.layers[15:]:
	layer.trainable = True

# looping of the layers in the model to show which ones are trainable
for layer in baseModel.layers:
	print("{}: {}".format(layer, layer.trainable))

# for the new changes to the model to take place, recompiling model with small learning rate

opt = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#train the model again, this time fine-tuning the final set of
# CONV layers along with our set of FC layers
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=20)
'''
# reset the testing generator and then use trained model
# to make predictions
print("*** evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
plot_training(H, 20, config.UNFROZEN_PLOT_PATH)
'''
# serialize the model to disk
model.save(config.MODEL_PATH)
