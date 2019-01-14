#!/usr/bin/python
## Python 3.x
import cv2, csv, sys, os
import numpy as np

def alignImages(image1, image2):

	# Convert images to grayscale
	im1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	im2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	# Detect ORB features and compute descriptors.
	orb = cv2.ORB_create(MAX_FEATURES)
	keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
	keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

	# Match features.
	matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	matches = matcher.match(descriptors1, descriptors2, None)

	# Sort matches by score
	matches.sort(key=lambda x: x.distance, reverse=False)

	# Remove not so good matches
	numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:numGoodMatches]

	# Draw top matches
	# = cv2.drawMatches(im1, keypoints1, image2, keypoints2, matches, None)
	#cv2.imwrite("matches.jpg", imMatches)

	# Extract location of good matches
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = keypoints1[match.queryIdx].pt
		points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography
	h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# Use homography
	height, width, channels = image2.shape

	im1Reg = cv2.warpPerspective(image1, h, (width, height))

	return im1Reg, h


#Some adjustable photo mapping params according to opencv examples
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

#Create list of touples for BRG and NIR photos
batch_photos = []

try:
	if(len(sys.argv) == 2):
		#Try to pen passed file and load it's contents
		with open(sys.argv[1]) as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			## TODO: Check if passed files are formatted correctly & accessible. Should be 2 columns, proper images from RGB and NIR cameras etc..
			for row in csvReader:
				batch_photos.append((row[0], row[1]))
		csvDataFile.close()
	else:
		print("Error: Please pass a csv file with the photos as a parameter.")
		sys.exit()
except IOError:
	print("Error: Import csv file does not exist.")
	sys.exit()


#If we have entries - lets so the magic
if len(batch_photos) > 0:
	for collection in batch_photos:

		# Read BASLER_POS/NIR image - first csv column
		refFilename = collection[0]
		print("Reading reference image (NIR): ", refFilename)
		imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

		# Read LUMENERA_POS/RGB image - second csv column
		imFilename = collection[1]
		print("Reading image to align (RGB): ", imFilename);  
		im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

		## Take NIR (first file) name as base for putput png naming and strip all directories
		filename = os.path.basename(refFilename)
		## Take file name and attach .png
		finalfilename = "FinalOutput/final_%s.png" % os.path.splitext(filename)[0]

		print("Aligning images into %s" % finalfilename)
		# Final image will be resotred in imReg. 
		# The estimated homography will be stored in h. 
		imReg, h = alignImages(im, imReference)

		'''
		#Splitting and Merging Image Channels
		b,g,r = cv2.split(imReg)

		cv2.imwrite('channel-B.jpg', b)
		cv2.imwrite('channel-R.jpg', r)
		cv2.imwrite('channel-G.jpg', g)
		'''

		# Write aligned image to disk. 
		#outFilename = "aligned.jpg"
		#print("Saving aligned image : ", outFilename); 
		#cv2.imwrite(outFilename, imReg)

		# Print estimated homography
		#print("Estimated homography : \n",  h)

		##Stack horizontally
		##vis = np.concatenate((imReference, imReg), axis=1)
		##cv2.imwrite('side-by-side.png', vis)

		#imstack = np.hstack(imReference, imReg)

		## Just do a quick and dirty simple merge of the images using 0.5+0.5 weight of the both RGB and NIR inputs..
		finalimg = cv2.addWeighted(imReg, 0.5, imReference, 0.5, 0)

		cv2.imwrite(finalfilename, finalimg)
else:
	print("No images found in the input csv file")
