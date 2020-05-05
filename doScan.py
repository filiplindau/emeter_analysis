import numpy as np
import time
import PyTango as pt

slitDeviceName=('b-v-gunlab-csdb-0:10000/g-b080003/opt/slit-01-x')
platDeviceName=('b-v-gunlab-csdb-0:10000/g-b080003/dia/tab-01-z')
camDeviceName=('b-v-gunlab-csdb-0:10000/lima/liveviewer/g-b080003-dia-scrn-em')

baseName='data/slitscan-M'

platPos=[250,300,350,400]

slitStart=10.0
slitStop=15.3
slitSpacing=0.3
slitPos=np.arange(slitStart, slitStop, slitSpacing)

numSlitImages=5

bkgPos=15.0
numBkg=5

slitDevice=pt.DeviceProxy(slitDeviceName)
camDevice=pt.DeviceProxy(camDeviceName)
platDevice=pt.DeviceProxy(platDeviceName)

def moveAxis(devId, setPos):
	devId.position = setPos
	while (abs(devId.position - setPos) > 0.007):
		time.sleep(0.1)

	#ugly fix
	time.sleep(2)


for curPlatPos in platPos:
	#move platform into position
	moveAxis(platDevice, curPlatPos)

	print("Platform in pos, recording at position " + str(curPlatPos))

	print("Doing background shot")
	# move into the specified background position
	moveAxis(slitDevice, bkgPos)
	for curImg in np.arange(numBkg):
		# record images
		bkgFileName=baseName + '-bkg-' + str(curPlatPos) + '-' + str(curImg)
		np.save(bkgFileName,camDevice.Image.astype(np.int16))
		tmpImgNum=camDevice.ImageCounter
		while (camDevice.ImageCounter - tmpImgNum < 1):
			time.sleep(0.5)
		print("Saved background image " + str(curImg) + " " + bkgFileName)

	for curSlitPos in slitPos:
		# Move slit into position
		moveAxis(slitDevice, curSlitPos)

		# record numslitimages at this position
		for curImg in np.arange(numSlitImages):
			dataFileName=baseName + '-' +str(curPlatPos) + '-' + str(curSlitPos) + '-' + str(curImg)
			np.save(dataFileName, camDevice.Image.astype(np.int16))
			tmpImgNum=camDevice.ImageCounter
			while (camDevice.ImageCounter - tmpImgNum < 1):
				time.sleep(0.5)
		
		print("Done with images at slitpos " + str(curSlitPos))
			

print("Done with all positions, exiting...")
