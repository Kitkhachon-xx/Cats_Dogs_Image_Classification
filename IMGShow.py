import matplotlib.pyplot as plt
import cv2
import pandas as pd
from glob import glob
from pathlib import Path

class ImageLoader:
# Resolve dataset paths relative to this script instead of the old Kaggle-style ../input layout
	base_dir = Path(__file__).resolve().parent
	catFiles = glob(str(base_dir / 'training_set' / 'training_set' / 'cats' / '*.jpg'))
	dogFiles = glob(str(base_dir / 'training_set' / 'training_set' / 'dogs' / '*.jpg'))

if not ImageLoader.catFiles:
	raise FileNotFoundError('No cat images found. Check training_set/training_set/cats/*.jpg')
if not ImageLoader.dogFiles:
	raise FileNotFoundError('No dog images found. Check training_set/training_set/dogs/*.jpg')


class IMGShow :
# Read one sample image two ways to compare how libraries load pixel data
	img_mp1 = plt.imread(ImageLoader.catFiles[0])
	img_mp2 = cv2.imread(ImageLoader.catFiles[0])
	print(len(ImageLoader.catFiles), 'cat images found.')
def IMGShowSample() :
# Inspect the resulting ndarray type from matplotlib's loader
	#print(img_mp1.flatten())
	pd.Series(IMGShow.img_mp1.flatten()).plot(kind='hist', bins=50, title='Pixel Value Distribution - Matplotlib Loader')
	#plt.show()
	plt.imshow(IMGShow.img_mp1)
	plt.title('Cat Image Loaded with Matplotlib')
	#plt.show()
	return IMGShowSample

def IMGShowLayer() : 
	fig, ax = plt.subplots(1, 3, figsize=(15, 5))
	ax[0].axis('off')
	ax[0].imshow(IMGShow.img_mp1[:,:,0], cmap='Reds')  # Convert BGR to RGB for correct color display
	ax[1].axis('off')
	ax[1].imshow(IMGShow.img_mp1[:, :, 1], cmap = 'Greens')
	ax[2].axis('off')
	ax[2].imshow(IMGShow.img_mp1[: , :, 2], cmap = 'Blues')
	plt.suptitle('Cat Image Loaded with Matplotlib - RGB Channels')
	#plt.show()

	fig, ax = plt.subplots(1, 3, figsize=(15, 5))
	ax[0].imshow(IMGShow.img_mp1)
	ax[0].set_title('Matplotlib Loader')
	ax[0].axis('off')
	ax[1].imshow(IMGShow.img_mp2)
	ax[1].set_title('OpenCV Loader')
	ax[1].axis('off')
	ax[2].imshow(cv2.cvtColor(IMGShow.img_mp2, cv2.COLOR_BGR2RGB))
	ax[2].set_title('OpenCV Loader With Color Correction')
	ax[2].axis('off')
	plt.suptitle('Cat Image Loaded with Different Libraries')
	#plt.show()
	return IMGShowLayer





