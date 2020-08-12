import cv2
import glob

img_size = (416,416)

for img_path in glob.glob("*.jpg"):
	print(img_path)
	img = cv2.imread(img_path)
	img_rsz = cv2.resize(img, img_size, interpolation = cv2.INTER_AREA)
	cv2.imwrite(img_path,img_rsz)