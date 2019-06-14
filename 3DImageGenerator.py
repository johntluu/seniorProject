#!/cm/shared/apps/virtualenv/csc/bin/python3.6
def combineRGBandDepth(rgbImage, depthImage):
    rgbdImage = []
    rgbdImage.append(rgbImage)
    rgbdImage.append(depthImage)
    return rgbImage
