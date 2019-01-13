## Python 3.x
import cv2, csv, sys, os
import numpy as np
 
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

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


#Create separate lists of photos for BRG and NIR
monochome_nir = []
color_rgb = []

try:
	#Try to pen passed file and load it's contents
	with open('batch_photo_paths.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		## TODO: Check if passed file is formatted correctly. Like only 2 columns, proper images from RGB and NIR cameras etc..
		## Files should also be accessable..
		for row in csvReader:
			monochome_nir.append(row[0])
			color_rgb.append(row[1])
	csvDataFile.close()

except IOError:
	print("Error: Import csv file does not exist.")
	sys.exit()


'''
# Read reference image
refFilename = "img_basler.1505292392366_.tiff"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read image to be aligned
imFilename = "img_lum.1505292339490_.jpeg"
print("Reading image to align : ", imFilename);  
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
'''

# Read BASLER_POS/NIR image
refFilename = monochome_nir[0] #"img_basler.1505292392366_.tiff"
print("Reading reference image (NIR): ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)


# Read LUMENERA_POS/RGB image
imFilename = color_rgb[0] #"img_lum.1505292339490_.jpeg"
print("Reading image to align (RGB): ", imFilename);  
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

## Take NIR (first file) name as base for putput png naming and strip all directories
filename = os.path.basename(refFilename)
## Take file name and attach .png
finalfilename = "FinalOutput/final_%s.png" % os.path.splitext(filename)[0]

print("Aligning images into %s" % finalfilename)
# Registered image will be resotred in imReg. 
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

