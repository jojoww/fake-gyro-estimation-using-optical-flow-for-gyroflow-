# Import numpy and OpenCV
import numpy as np
import cv2
import os


############ Start of config ############

# Copy the camera matrix from the lens profile you are using in gyroflow
cameraMatrix = [
      [
        1836.9922183665276,
        0.0,
        1920.41209963424
      ],
      [
        0.0,
        1842.3851575279864,
        1074.5706377526926
      ],
      [
        0.0,
        0.0,
        1.0
      ]
    ]

distortionCoeffs = [
      0.38299220710571275,
      -0.17675639024960604,
      0.7856650933984539,
      -0.5395566240889261
    ]
# The scale can be used for fixing the result. If the detected shifts
# are too small (e.g. camera movements are not possible to be eliminated
# completely), a correction with scale > 1 might help (e.g. 1.05)
scale = 1.0

internalImageWidth = 1280
############ End of config ############

cameraMatrix = np.array(cameraMatrix)
distortionCoeffs = np.array(distortionCoeffs)

def preprocessImage(image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fisheye.undistortImage(image, cameraMatrix, distortionCoeffs, np.eye(3), newCameraMatrix)
    image = cv2.resize(image, (width, height))
    return image

def getFovAndCanvasDistance(videoWith, internalImageWidth, cameraMatrix):
    fov = 2 * np.arctan2(videoWith, 2 * cameraMatrix[1][1])
    cameraDistance = (internalImageWidth / 2) * \
        np.sin(np.pi / 2 - fov / 2) / np.sin(fov / 2)
    print("Assuming horizontal FOV", np.rad2deg(fov), "degree")
    return fov, cameraDistance


def vectorAngle(vec1, vec2):
    unitVector1 = vec1 / np.linalg.norm(vec1)
    unitVector2 = vec2 / np.linalg.norm(vec2)
    dotProduct = np.dot(unitVector1, unitVector2)
    angle = np.arccos(dotProduct)
    return angle


def askUserForFiles():
    allFiles = next(os.walk("."), (None, None, []))[2]
    allFilesFiltered = [f for f in allFiles if not f.endswith(
        "py") and not f.endswith("csv") and "stabilized" not in f]
    print(
        "\n".join(allFilesFiltered),
        "\nFound the abovementioned files. Which one(s) should be processed?")
    print("Enter file name, or file name part that matches multiple files, e.g. 'MP4' to process all MP4 files.")
    filePattern = input()
    return [f for f in allFilesFiltered if filePattern in f]


def getCsvHeader(fps):
    # So far, this is just the example file from the gyroflow documentation,
    # but with proper fps
    return f"""GYROFLOW IMU LOG
        version,1.3
        id,custom_logger_name
        orientation,xyz
        note,development_test
        fwversion,FIRMWARE_0.1.0
        timestamp,1644159993
        vendor,potatocam
        videofilename,videofilename.mp4
        lensprofile,potatocam/potatocam_mark1_prime_7_5mm_4k
        lens_info,wide
        frame_readout_time,16.23
        frame_readout_direction,0
        tscale,{1/fps}
        gscale,1.0
        ascale,1.0
        t,gx,gy,gz
        """


def getCameraShiftByRegistration(previousImage, currentImage, width, height):
    # - Just an incomplete draft -
    # Find size of image1
    sz = previousImage.shape
    
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION
    #warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    
    # Specify the number of iterations.
    number_of_iterations = 500
    
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-5
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (previousImage, currentImage,warp_matrix, warp_mode, criteria)
    
    #if warp_mode == cv2.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective (currentImage, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # else :
    # Use warpAffine for Translation, Euclidean and Affine
    # im2_aligned = cv2.warpAffine(currentImage, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return getCameraShiftLK(im2_aligned, currentImage, width, height)


def getCameraShiftLK(previousImage, currentImage, width, height):
        # Detect feature points in previous frame
    previousPoints = cv2.goodFeaturesToTrack(previousImage,
                                        maxCorners=200,
                                        qualityLevel=0.01,
                                        minDistance=30,
                                        blockSize=3)


    # Calculate optical flow (i.e. track feature points)
    currentPoints, status, err = cv2.calcOpticalFlowPyrLK(
        previousImage, currentImage, previousPoints, None)

    # Sanity check
    assert previousPoints.shape == currentPoints.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    previousPoints = previousPoints[idx]
    currentPoints = currentPoints[idx]

    # Shift the points, since gyroflow looks at the center of an image;
    # inverse y axis
    previousPointsCorrected = previousPoints - [width / 2, height / 2]
    previousPointsCorrected *= [1, -1]
    currentPointsCorrected = currentPoints - [width / 2, height / 2]
    currentPointsCorrected *= [1, -1]

    # Find transformation matrix
    m = cv2.estimateAffinePartial2D(previousPointsCorrected, currentPointsCorrected)[0]
    minimumTrackedFeatures = 5
    if len(previousPoints) > minimumTrackedFeatures and len(
            currentPoints) > minimumTrackedFeatures:
        # Extract translation
        dx = m[0, 2]
        dy = m[1, 2]

        # Extract rotation angle
        da = np.arctan2(m[1, 0], m[0, 0])
    else:
        dx, dy, da = 0, 0, 0
    return dx, dy, da


for fileName in askUserForFiles():
    print("Processing file", fileName)
    baseFileName = ".".join(fileName.split(".")[0:-1])

    fileNameGyro = baseFileName + '.gcsv'
    fileNameTrajectory = baseFileName + ".transforms.npy"

    cap = cv2.VideoCapture(fileName)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distortionCoeffs, size, 1, size)
    newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distortionCoeffs, size, np.eye(3))

    numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT) /
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH) *
        internalImageWidth)
    fov, cameraDistance = getFovAndCanvasDistance(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), internalImageWidth, cameraMatrix)
    width = int(internalImageWidth)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Analyze the video, if we don't have a trajectory file already
    if not os.path.isfile(fileNameTrajectory):
        _, image = cap.read()
        previousImage = preprocessImage(image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)

        # Get transformations
        transforms = np.zeros((numberOfFrames - 1, 3), np.float32)
        index = 0
        # Analyze the video, if we don't have a trajectory file already
        for _ in range(numberOfFrames - 2):
             # Read next frame
            success, image = cap.read()
            if image is None:
                # index -= 1
                print("Had to skip frame", index)
                continue

            print(f"Frame {index} of {numberOfFrames}")
            currentImage = preprocessImage(image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)
            dx, dy, da = getCameraShiftLK(previousImage, currentImage, width, height)

            # Store transformation
            transforms[index] = [dx, dy, da]

            previousImage = currentImage

            index += 1
        np.save(fileNameTrajectory, transforms)
    cap.release()

    csvOut = getCsvHeader(fps)
    transforms = np.load(fileNameTrajectory)

    for i, step in enumerate(transforms):
        dx, dy, da = step
        da = da * fps

        # Get vectors looking at the "shift targets"
        xDirection = np.array([dx, cameraDistance])
        yDirection = np.array([dy, cameraDistance])
        axis = np.array([0, 1])

        # Find out the angle you need in order to reach the "shift target"
        dxRotation = -1 * np.sign(dx) * \
            vectorAngle(xDirection, axis) * fps * scale
        dyRotation = -1 * np.sign(dy) * \
            vectorAngle(yDirection, axis) * fps * scale

        # Not sure about the order of rotations. In case gyroflow has a specific order,
        # we could change it this way. I think so.
        # from scipy.spatial.transform import Rotation
        # rot = Rotation.from_euler('xyz', [dxRotation, dyRotation, da], degrees=False)
        # # Convert to quaternions and print
        # rot_adjusted = rot.as_euler('zyx')
        # dxRotation, dyRotation, da = rot_adjusted

        csvOut += f"{i},{dxRotation},{dyRotation},{da}\n"
    f = open(fileNameGyro, "w")
    f.write(csvOut)
    f.close()
    print(
        "Done with",
        fileName,
        f"- in gyroflow, use a manual sync point with {(1000/fps):.2f}ms")
