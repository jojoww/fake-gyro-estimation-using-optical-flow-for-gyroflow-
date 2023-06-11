import numpy as np
import cv2
import os


############ Start of config ############

# Copy the camera matrix from the lens profile you are using in gyroflow
cameraMatrix = [
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
]

# The scale can be used for fixing the result. If the detected shifts
# are too small (e.g. camera movements are not possible to be eliminated
# completely), a correction with scale > 1 might help (e.g. 1.05)
scale = 1.0

internalImageWidth = 1280
minimumTrackedFeatures = 5
############ End of config ############


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


for fileName in askUserForFiles():
    print("Processing file", fileName)

    fileNameGyro = fileName + '.gcsv'
    fileNameTrajectory = fileName + ".transforms.npy"

    cap = cv2.VideoCapture(fileName)
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
        _, prev = cap.read()
        prev = cv2.resize(prev, (width, height))
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # Get transformations
        transforms = np.zeros((numberOfFrames - 1, 3), np.float32)
        for i in range(numberOfFrames - 2):
            # Detect feature points in previous frame
            prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                               maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30,
                                               blockSize=3)

            # Read next frame
            success, curr = cap.read()
            curr = cv2.resize(curr, (width, height))
            if not success:
                break

            # Convert to grayscale
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow (i.e. track feature points)
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None)

            # Sanity check
            assert prev_pts.shape == curr_pts.shape

            # Filter only valid points
            idx = np.where(status == 1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Shift the points, since gyroflow looks at the center of an image;
            # inverse y axis
            prev_pts_shifted = prev_pts - [width / 2, height / 2]
            prev_pts_shifted *= [1, -1]
            curr_pts_shifted = curr_pts - [width / 2, height / 2]
            curr_pts_shifted *= [1, -1]

            # Find transformation matrix
            m = cv2.estimateAffinePartial2D(prev_pts_shifted, curr_pts_shifted)[
                0]  # will only work with OpenCV-3 or less

            if len(prev_pts) > minimumTrackedFeatures and len(
                    curr_pts) > minimumTrackedFeatures:
                # Extract translation
                dx = m[0, 2]
                dy = m[1, 2]

                # Extract rotation angle
                da = np.arctan2(m[1, 0], m[0, 0])
            else:
                dx, dy, da = 0, 0, 0

            # Store transformation
            transforms[i] = [dx, dy, da]

            prev_gray = curr_gray

            print(
                f"Frame {i} of {numberOfFrames}. Tracked points : {len(prev_pts)}")
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
        # rot = Rotation.from_euler('xyz', [dx, dy, da], degrees=False)
        # Convert to quaternions and print
        # rot_adjusted = rot.as_euler('zxy')
        # dx, dy, da = rot_adjusted

        csvOut += f"{i},{dxRotation},{dyRotation},{da}\n"
    f = open(fileNameGyro, "w")
    f.write(csvOut)
    f.close()
    print(
        "Done with",
        fileName,
        f"- in gyroflow, use a manual sync point with {(1000/fps):.2f}ms")
