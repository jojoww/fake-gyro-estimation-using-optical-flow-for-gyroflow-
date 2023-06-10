import numpy as np
import cv2
import os


############ Start of config ############


# I am very sorry for the following lines. These scales convert the pixel-based
# shift of optical flow to an angular shift. Because I didn't care about the maths, 
# I just figured it out "experimentally". The value "4.95", for example, corresponds
# to a full frame 16mm wide angle. The value 3.0 corresponds to 24mm full frame but
# with a video crop of factor 1.1
#
# This value depends on crop, lens profile and the internal scale for feature
# detection (which I set to 1280 pixels below).
# 
# I am very sure you can determine this by analyzing the actual field of view, e.g.
# from gyroflow's lens profiles.
scaleX = 8.4
scaleY = scaleX
scaleA = 1.0
############ End of config ############

allFiles = next(os.walk("."), (None, None, []))[2]
allFilesFiltered = [f for f in allFiles if not f.endswith("py") and not f.endswith("csv") and not "stabilized" in f]
print("\n".join(allFilesFiltered), "\nFound the abovementioned files. Which one(s) should be processed?")
print("Enter file name, or file name part that matches multiple files, e.g. 'MP4' to process all MP4 files.")
filePattern = input()
filesToProcess = [f for f in allFilesFiltered if filePattern in f]

for fileName in filesToProcess:
    print("Start preparing file", fileName)

    internalImageWidth = 1280
    fileNameGyro  = fileName + '.gcsv'
    fileNameTrajectory  = fileName + ".transforms.npy"

    cap = cv2.VideoCapture(fileName) 
    numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / cap.get(cv2.CAP_PROP_FRAME_WIDTH) * internalImageWidth)
    width = int(internalImageWidth)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Analyze the video, if we don't have a trajectory file already
    if not os.path.isfile(fileNameTrajectory):  
        _, prev = cap.read() 
        prev = cv2.resize(prev, (width, height))
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

        # Get transformations 
        transforms = np.zeros((numberOfFrames - 1, 3), np.float32) 
        for i in range(numberOfFrames-2):
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
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

            # Sanity check
            assert prev_pts.shape == curr_pts.shape 

            # Filter only valid points
            idx = np.where(status==1)[0]
            prev_pts = prev_pts[idx]
            curr_pts = curr_pts[idx]

            # Shift the points, since gyroflow looks at the center of an image
            prev_pts_shifted = prev_pts - [width / 2, height / 2]
            prev_pts_shifted *= [1, -1]
            curr_pts_shifted = curr_pts - [width / 2, height / 2]
            curr_pts_shifted *= [1, -1]

            #Find transformation matrix
            m = cv2.estimateAffinePartial2D(prev_pts_shifted, curr_pts_shifted)[0] #will only work with OpenCV-3 or less
            print(m)


            if len(prev_pts) > 5 and len(curr_pts) > 5:
                # Extract translation
                dx = m[0,2]
                dy = m[1,2]

                # Extract rotation angle
                da = np.arctan2(m[1,0], m[0,0])
            else:
                dx, dy, da = 0, 0, 0
        
            # Store transformation
            transforms[i] = [dx,dy,da]
            
            prev_gray = curr_gray

            print(f"Frame {i} of {numberOfFrames}. Tracked points : {len(prev_pts)}")
        np.save(fileNameTrajectory, transforms)
    cap.release()


    csvOut = f"""GYROFLOW IMU LOG
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
    transforms = np.load(fileNameTrajectory)

    for i, step in enumerate(transforms):
        dx, dy, da = step
        dx *= scaleX / 100
        dy *= scaleY / 100
        da *= fps * scaleA


        # Not sure about the order of rotations. In case gyroflow has a specific order,
        # we could change it this way. I think so.
        # from scipy.spatial.transform import Rotation
        # rot = Rotation.from_euler('xyz', [dx, dy, da], degrees=False)
        # Convert to quaternions and print
        # rot_adjusted = rot.as_euler('zxy')
        # dx, dy, da = rot_adjusted
        
        csvOut += f"{i},{-dx},{-dy},{da}\n"
    f = open(fileNameGyro, "w")
    f.write(csvOut)
    f.close()
    print("Done with", fileName, f"- in gyroflow, use a manual sync point with {(1000/fps):.2f}ms")


