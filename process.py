## Python 3.x
import cv2, csv, sys
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


# Read reference image
refFilename = "img_basler.1505292392366.tiff"

print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# Read image to be aligned
imFilename = "img_lum.1505292339490.jpeg"

print("Reading image to align : ", imFilename);  
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

print("Aligning images ...")
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
outFilename = "aligned.jpg"

print("Saving aligned image : ", outFilename); 
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n",  h)

##Stack horizontally
##vis = np.concatenate((imReference, imReg), axis=1)
##cv2.imwrite('side-by-side.png', vis)

#imstack = np.hstack(imReference, imReg)

final = cv2.addWeighted(imReg, 0.5, imReference, 0.5, 0)

cv2.imwrite('final.png', final)

