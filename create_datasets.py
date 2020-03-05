import os
import shutil
import random
from configurations import config

try:
	os.mkdir(os.path.join(config.BASE_PATH, config.VALIDATION))
	os.mkdir(os.path.join(config.BASE_PATH, config.TEST))
except:
	print("dumbfuck")
	exit()

class_paths = os.listdir(os.path.join(config.BASE_PATH, config.TRAIN))
class_paths = [os.path.join(config.BASE_PATH, config.TRAIN) + os.path.sep + class_path for class_path in class_paths]
src_path = os.path.join(config.BASE_PATH, config.TRAIN)

for class_path in class_paths:
	img_paths = [class_path + os.path.sep + img_path for img_path in os.listdir(class_path)]

	random.shuffle(img_paths)
	test_imgs = img_paths[:int(len(img_paths) * 0.2)]
	val_imgs = img_paths[-int(len(img_paths) * 0.2):]

	for test_img in test_imgs:
		print (test_img)
		shutil.move(test_img, test_img.replace('training', 'test'))

	for val_img in val_imgs:
		print (val_img)
		shutil.move(val_img, val_img.replace('training', 'validation'))
