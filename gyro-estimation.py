# Import numpy and OpenCV
import numpy as np
import cv2
import os
from exif import Image

############ Start of config ############
tracker = "doubleflow"  # matcher, flow, doubleflow, registration

# Copy the camera matrix from the lens profile you are using in gyroflow
lensProfiles = {
    # Add as many profiles as you like, in case video file has enough metadata
    # for a unique identifier (e.g. lens and camera name)
    "3840_2160_59_Canon EOS R8_16.0_*_RF16mm F2.8 STM": {
        "camera_matrix": [
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
        ],
        "distortion_coeffs": [
            0.38299220710571275,
            -0.17675639024960604,
            0.7856650933984539,
            -0.5395566240889261
        ]
    },
    # ... and this is the fallback if no profile matches:
    "default": {
        "camera_matrix": [
            [
                5801.606439877379,
                0.0,
                1930.5760969035825
            ],
            [
                0.0,
                5810.509627755767,
                1112.9812445517732
            ],
            [
                0.0,
                0.0,
                1.0
            ]
        ],
        "distortion_coeffs": [
            0.27548726435520854,
            -2.1102239947731887,
            25.6426432549478,
            -99.77349159535086
        ]
    },
}

# The scale can be used for fixing the result. If the detected shifts
# are too small (e.g. camera movements are not possible to be eliminated
# completely), a correction with scale > 1 might help (e.g. 1.05)
scale = 1.0
internalImageWidth = 1280
############ End of config ############


def preprocessImage(image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.fisheye.undistortImage(
        image, cameraMatrix, distortionCoeffs, np.eye(3), newCameraMatrix)
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


def pairedPointsToPixelShifts(points1, points2, width, height):
    points1 = points1 - [width / 2, height / 2]
    points1 *= [1, -1]
    points2 = points2 - [width / 2, height / 2]
    points2 *= [1, -1]
    # Find homography
    m = cv2.estimateAffinePartial2D(points1, points2)[0]
    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])
    return dx, dy, da


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
    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    numberOfIterations = 500
    terminationEps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                numberOfIterations, terminationEps)
    (_, m) = cv2.findTransformECC(previousImage,
                                  currentImage, warp_matrix, warp_mode, criteria)

    # Create some sample points that we can use to create the rotation matrix in
    # "shifted image space"
    points1 = np.float32(np.array([[
        [0, 0],
        [int(width / 2), 0],
        [width, 0],
        [int(width / 2), int(height/2)],
        [width, int(height/2)],
        [int(width / 2), height],
        [width, height],
    ]]))

    points2 = cv2.perspectiveTransform(points1, m)
    return pairedPointsToPixelShifts(points1, points2, width, height)


def getCameraShiftFeature(previousImage, currentImage, width, height):
    maxFeatures = 500
    topMatchesToUsePercentage = 0.15
    orb = cv2.ORB_create(maxFeatures)
    keypoints1, descriptors1 = orb.detectAndCompute(previousImage, None)
    keypoints2, descriptors2 = orb.detectAndCompute(currentImage, None)
    matcher = cv2.DescriptorMatcher_create(
        cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * topMatchesToUsePercentage)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(previousImage, keypoints1, currentImage, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    minimumTrackedFeatures = 5
    print(f"Found {len(matches)} features")
    if len(matches) > minimumTrackedFeatures:
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = np.array(keypoints1[match.queryIdx].pt)
            points2[i, :] = np.array(keypoints2[match.trainIdx].pt)

        return pairedPointsToPixelShifts(points1, points2, width, height)
    else:
        return 0, 0, 0


def getCameraShiftLK(previousImage, currentImage, width, height):
    # Detect feature points in previous frame
    borderHeight = int(height/5)
    borderWidth = int(height/5)
    edgeMask = np.zeros((height, width), dtype=np.uint8)
    edgeMask[borderHeight:(height-borderHeight),
             borderWidth:(width-borderWidth)] = 1
    points1 = cv2.goodFeaturesToTrack(previousImage,
                                      maxCorners=200,
                                      qualityLevel=0.01,
                                      minDistance=30,
                                      blockSize=3,
                                      mask=edgeMask)

    points2, status, _ = cv2.calcOpticalFlowPyrLK(
        previousImage, currentImage, points1, None)
    assert points1.shape == points2.shape
    idx = np.where(status == 1)[0]
    points1 = points1[idx]
    points2 = points2[idx]
    print(f"Found {len(points1)} features")
    minimumTrackedFeatures = 5
    if len(points1) > minimumTrackedFeatures and len(points2) > minimumTrackedFeatures:
        return pairedPointsToPixelShifts(points1, points2, width, height)
    else:
        return 0, 0, 0


def getExifFieldIfExists(video, field):
    if field in video.list_all():
        return video[field]
    return "*"


def getCameraIdentifier(fileName, cap):
    identifierElements = [
        str(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        str(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        str(int(cap.get(cv2.CAP_PROP_FPS))),
    ]
    try:
        with open(fileName, 'rb') as videoFile:
            video = Image(videoFile)
            if video.has_exif:
                identifierElements += [
                    getExifFieldIfExists(video, "model"),
                    str(getExifFieldIfExists(video, "focal_length")),
                    getExifFieldIfExists(video, "lens_make"),
                    getExifFieldIfExists(video, "lens_model"),
                ]
    except Exception as e:
        print("Could not read extended camera information.")
        # print("Error:", e)
    return "_".join(identifierElements)


def loadLensProfile(lensProfiles, identifier):
    print("Load profile", identifier)
    cameraMatrix = np.array(lensProfiles[identifier]["camera_matrix"])
    distortionCoeffs = np.array(lensProfiles[identifier]["distortion_coeffs"])
    return cameraMatrix, distortionCoeffs


for fileName in askUserForFiles():
    print("Processing file", fileName)
    baseFileName = ".".join(fileName.split(".")[0:-1])

    fileNameGyro = baseFileName + '.gcsv'
    fileNameTrajectory = baseFileName + ".transforms.npy"

    cap = cv2.VideoCapture(fileName)
    cameraIdentifier = getCameraIdentifier(fileName, cap)
    if cameraIdentifier in lensProfiles:
        cameraMatrix, distortionCoeffs = loadLensProfile(
            lensProfiles, cameraIdentifier)
    else:
        cameraMatrix, distortionCoeffs = loadLensProfile(
            lensProfiles, "default")

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, distortionCoeffs, size, 1, size)
    newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cameraMatrix, distortionCoeffs, size, np.eye(3))

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
        previousImage = preprocessImage(
            image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)

        # Get transformations
        transforms = np.zeros((numberOfFrames, 3), np.float32)
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
            currentImage = preprocessImage(
                image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)
            if tracker == "flow":
                dx, dy, da = getCameraShiftLK(
                    previousImage, currentImage, width, height)
            elif tracker == "doubleflow":
                dx1, dy1, da1 = getCameraShiftLK(
                    previousImage, currentImage, width, height)
                dx2, dy2, da2 = getCameraShiftLK(
                    currentImage, previousImage, width, height)
                dx = (dx1 - dx2) / 2.
                dy = (dy1 - dy2) / 2.
                da = (da1 - da2) / 2.
            elif tracker == "matcher":
                dx, dy, da = getCameraShiftFeature(
                    previousImage, currentImage, width, height)
            else:
                dx, dy, da = getCameraShiftByRegistration(
                    previousImage, currentImage, width, height)

            # Store transformation
            transforms[index + 1] = [dx, dy, da]

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

        csvOut += f"{i},{dxRotation},{dyRotation},{da}\n"
    f = open(fileNameGyro, "w")
    f.write(csvOut)
    f.close()
    if cameraIdentifier is not None and cameraIdentifier not in lensProfiles:
        print("\nPS: The script can auto-assign this profile to all videos",
              "with similar settings. Add it to 'lensProfiles'",
              f"under the name \n'{cameraIdentifier}'\n")

    print(
        "Done with",
        fileName,
        f"- in gyroflow, use a manual sync point with {(1000/fps):.2f}ms")
