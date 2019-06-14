#!/cm/shared/apps/virtualenv/csc/bin/python3.6
import os
import random
import cv2
from sys import exit
from depthImageGenerator import initialize, generate_depth_image

def load_all_datasets(directory):
    trainingSet, validationSet, testingSet, depthImagesTrainingSet, depthImagesValidationSet, depthImagesTestingSet = split_dataset(directory)

    x_train, y_train = load_dataset(trainingSet)
    x_validation, y_validation = load_dataset(validationSet)
    x_test, y_test = load_dataset(testingSet)

    return x_train, y_train, x_validation, y_validation, x_test, y_test, depthImagesTrainingSet, depthImagesValidationSet, depthImagesTestingSet

def split_dataset(directory):
    rootDir = directory
    allFilenames = []
    # depthImages = []
    # notDetectedFaces = []
    # foundFaces = []
    directoryData = []
    trainingSet = []
    validationSet = []
    testingSet = []
    depthImagesTraining = []
    depthImagesValidation = []
    depthImagesTesting = []
    
    prn = initialize()

    for root, dirs, files in os.walk(rootDir):
        # filenames = []

        # does this second
        # for name in files:
        #     print("\nFilepath: {}".format(os.path.join(root, name)))
        #     print(os.path.join(root, name))
        #     filenames.append(os.path.join(root, name))
            
        # does this first
        for name in dirs:
            print("\nDirectory: {}".format(os.path.join(root, name)))
            print(os.path.join(root, name))
            depthImages, validPaths = generate_depth_image(os.path.join(root, name), prn)
            directoryData.append([validPaths, depthImages])
            # curDirDepthImgs, notDetectedPaths = generate_depth_image(os.path.join(root, name), prn)
            # depthImages.append(curDirDepthImgs)
            # notDetectedFaces.append(notDetectedPaths)
            # if (len(filenames) != 0): 
                # allFilenames.append(filenames)
                # directoryData.append([filenames, curDirDepthImgs, notDetectedPaths])

    # for i in range(len(directoryData)):
    #     finalFilenames = []
    #     print("\nPrinting paths of images with no detected faces:")
    #     for fn in directoryData[i][2]:
    #         print(fn)

    #     print("\nNumber of non-detected faces: {}\n".format(len(directoryData[i][2])))

    #     for j in range(len(directoryData[i][0])):
    #         if (directoryData[i][0][j] not in directoryData[i][2]):
    #             print("Appending {}".format(directoryData[i][0][j]))
    #             finalFilenames.append(directoryData[i][0][j])
    #         else:
    #             print("Not Appending {}".format(directoryData[i][0][j]))
                
    #     directoryData[i].append(finalFilenames)




    # for j in range(len(allFilenames)):
    #     print("\nPrinting paths of images with no detected faces:")
    #     for fn in notDetectedFaces[j]:
    #         print(fn)


    #     print("\nNumber of non detected faces: {}\n".format(len(notDetectedFaces[j])))
    #     if (len(allFilenames[j]) != 0):
    #         finalFilenames = []
    #         for i in range(len(allFilenames[j])):
    #             print("\nCurrent File: {}".format(allFilenames[j][i]))
    #             if (allFilenames[j][i] not in notDetectedFaces[j]):
    #                 print("Appending {}".format(allFilenames[j][i]))
    #                 finalFilenames.append(allFilenames[j][i])
    #             else:
    #                 print("Not Appending {}".format(allFilenames[j][i]))
    #             foundFaces.append(finalFilenames)


    # print("\nNumber of Detected Faces Directories: {}".format(len(foundFaces))) 
    print("\nNumber of Directories: {}\n".format(len(directoryData)))

    for i in range(len(directoryData)):
        print("Directory {}:".format(i))
        # print("Total Faces: {}".format(len(allFilenames[i])))
        # print("Not-Detected Faces: {}".format(len(directoryData[i][2])))
        print("Num Detected Faces: {}".format(len(directoryData[i][0])))
        for j in range(len(directoryData[i][0])):
            print(directoryData[i][0][j])
        print("Num Depth Images: {}\n".format(len(directoryData[i][1])))

    for i in range(len(directoryData)):
        curFilenames = directoryData[i][0]
        curDepthImages = directoryData[i][1]

        random.seed(100)
        random.shuffle(curFilenames)
        random.shuffle(curDepthImages)

        if (len(curDepthImages) != len(curFilenames)):
            print("Number of depth images: {}".format(len(curDepthImages)))
            print("Number of rgb images: {}".format(len(curFilenames)))
            print("number of depth images aren't equal to number of images")
            exit(2)

        section1 = int(0.8 * len(curFilenames))
        section2 = int(0.9 * len(curFilenames))
        
        train_imgs = curFilenames[:section1]
        validation_imgs = curFilenames[section1:section2]
        test_imgs = curFilenames[section2:]

        trainingSet.append(train_imgs)
        validationSet.append(validation_imgs)
        testingSet.append(test_imgs)

        train_imgs_depth = curDepthImages[:section1]
        validation_imgs_depth = curDepthImages[section1:section2]
        test_imgs_depth = curDepthImages[section2:]

        depthImagesTraining += train_imgs_depth
        depthImagesValidation += validation_imgs_depth
        depthImagesTesting += test_imgs_depth

    return trainingSet, validationSet, testingSet, depthImagesTraining, depthImagesValidation, depthImagesTesting

def load_dataset(dataset):
    x = [] # images
    y = [] # labels
    label = 0

    # dimensions
    width = 128
    height = 128
    channels = 3

    for filenames in dataset:
        for filename in filenames:
            x.append(cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR), (width, height), interpolation = cv2.INTER_CUBIC))
            y.append(label)
        label += 1

    return x, y
