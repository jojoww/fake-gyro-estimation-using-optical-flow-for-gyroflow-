# Import numpy and OpenCV
import numpy as np
import cv2
import os
from exif import Image

############ Start of config ############
tracker = "doubleflow"  # Choose from: matcher, flow, doubleflow, registration

# Copy the camera matrix and distortion coefficients from the lens profile 
# you are using in gyroflow
lensProfiles = {
    # Add as many profiles as you like, in case video file has enough metadata
    # for a unique identifier (e.g. lens and camera name). The script will tell
    # you the identifier that you should use once you have processed a video 
    # with "default" settings.
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
    # And this is the fallback if no profile matches. If your video doesn't
    # contain sufficient metadata, put your reuqired profiles here to the
    # default slot.
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
# completely), a correction with scale > 1 might help (e.g. 1.05).
# However, if you have a good lens profile, you should NOT USE THE SCALE.
scale = 1.0

# The size to which images are downsampled before motion detection. 
# Smaller resolution might lead to faster processing, but smaller
# details are not considered for motion detection.
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
    # Get average point. We assume the calculcated translation is optimum for this point.
    meanPoint = np.mean(points1, 0)
    return dx, dy, da, meanPoint[0, 0], meanPoint[0, 1]


def askUserForFiles():
    allFiles = next(os.walk("."), (None, None, []))[2]
    allFilesFiltered = [f for f in allFiles if not f.endswith(
        "py") and not f.endswith("csv") and "stabilized" not in f]
    print("Found the following files:")
    print(
        "\n".join(["   " + f for f in allFilesFiltered]),
        "\nWhich file(s) of the current folder should be processed?")
    print("Enter the file name, or just a file name part that matches multiple files,"
          "e.g. 'MP4' to process all MP4 files.")
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
    minimumTrackedFeatures = 5
    print(f"    Found {len(matches)} features")
    if len(matches) > minimumTrackedFeatures:
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = np.array(keypoints1[match.queryIdx].pt)
            points2[i, :] = np.array(keypoints2[match.trainIdx].pt)
        return pairedPointsToPixelShifts(points1, points2, width, height)
    else:
        return 0, 0, 0, 0, 0


def getCameraShiftLK(previousImage, currentImage, width, height):
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
    # Extract location of good matches
    idx = np.where(status == 1)[0]
    points1 = points1[idx]
    points2 = points2[idx]
    print(f"    Found {len(points1)} features")
    minimumTrackedFeatures = 5
    if len(points1) > minimumTrackedFeatures and len(points2) > minimumTrackedFeatures:
        return pairedPointsToPixelShifts(points1, points2, width, height)
    else:
        return 0, 0, 0, 0, 0


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
        pass
        # print("Error:", e)
    return "_".join(identifierElements)


def loadLensProfile(lensProfiles, identifier):
    print("Load profile", identifier)
    cameraMatrix = np.array(lensProfiles[identifier]["camera_matrix"])
    distortionCoeffs = np.array(lensProfiles[identifier]["distortion_coeffs"])
    return cameraMatrix, distortionCoeffs


for fileName in askUserForFiles():
    print("-" * 80)
    print("Start processing file", fileName)
    fileNameGyro = ".".join(fileName.split(".")[0:-1]) + '.gcsv'

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
    newCameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        cameraMatrix, distortionCoeffs, size, np.eye(3))

    numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(size[1] / size[0] * internalImageWidth)
    width = int(internalImageWidth)
    fov, cameraDistance = getFovAndCanvasDistance(
        size[0], internalImageWidth, cameraMatrix)
    fps = cap.get(cv2.CAP_PROP_FPS)

    _, image = cap.read()
    previousImage = preprocessImage(
        image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)

    transforms = np.zeros((numberOfFrames, 5), np.float32)
    index = 0
    for _ in range(numberOfFrames - 2):
        success, image = cap.read()
        if image is None:
            print("   Info: could not read frame", index, " - skip it")
            continue

        print(f"   Frame {index} of {numberOfFrames}")
        currentImage = preprocessImage(
            image, width, height, cameraMatrix, distortionCoeffs, newCameraMatrix)
        if tracker == "flow":
            dx, dy, da, meanX, meanY = getCameraShiftLK(
                previousImage, currentImage, width, height)
        elif tracker == "doubleflow":
            dx1, dy1, da1, meanX, meanY = getCameraShiftLK(
                previousImage, currentImage, width, height)
            dx2, dy2, da2, _, _ = getCameraShiftLK(
                currentImage, previousImage, width, height)
            dx = (dx1 - dx2) / 2.
            dy = (dy1 - dy2) / 2.
            da = (da1 - da2) / 2.
        elif tracker == "matcher":
            dx, dy, da, meanX, meanY = getCameraShiftFeature(
                previousImage, currentImage, width, height)
        else:
            dx, dy, da, meanX, meanY = getCameraShiftByRegistration(
                previousImage, currentImage, width, height)

        transforms[index + 1] = [dx, dy, da, meanX, meanY]
        previousImage = currentImage
        index += 1
    cap.release()
    csvOut = getCsvHeader(fps)

    for i, step in enumerate(transforms):
        dx, dy, da, meanX, meanY = step
        da = da * fps

        # Get vectors looking at the "shift target"
        xDirection = np.array([meanX + dx, meanY, cameraDistance])
        yDirection = np.array([meanX, meanY + dy, cameraDistance])
        axis = np.array([meanX, meanY, cameraDistance])

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
              "with similar settings, at least if the following identifier",
              "is somewhat unique. Add it to 'lensProfiles'",
              f"under the name \n'{cameraIdentifier}'\n")

    print("Done with", fileName)
