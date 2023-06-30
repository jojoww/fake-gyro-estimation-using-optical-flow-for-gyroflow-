# Estimate camera rotation using optical flow
The goal of this tool is to create a gcsv file that can be ingested by gyroflow in order to stabilize footage for which no gyro data was recorded.

## The general idea
The video is analyzed using optical flow - a method to calculate the translation and rotation of image content. 
The output of optical flow is a shift in pixel units. This script converts the pixel-based measure into angles, for which it needs information about the camera properties (e.g. field of view).

## Limitations
Optical flow just analyzes the image but it has no idea what is actual camera movement, what is movement of the videographer and what is motion of objects in the scene.
For example, a video filmed out of a moving car, or a video full of moving subjects, would probably cause a wrong motion estimation. 

## Instructions
1. Download and install python 3
2. Install required packages. Open a terminal and run "python -mpip install opencv-python", "python -mpip install exif" and "python -mpip install numpy"
3. Put the script into the folder of the video files. Open this script in a text editor and replace "camera_matrix" with the one that you find in the lens profile that you use in gyroflow
4. Execute the script by calling "python gyro-estimation.py". Now enter the filename of the video you want to stabilize
5. Wait until it's done
6. Open Gyroflow, open your video file, open the lens profile, open the gcsv file as motion data

## Finetuning
So far, the quality of the stabilization is not "optimum" (even considering the limitations mentioned above). 
Probably there is still some mathematical error in the script. 

To overcome this, you can edit the script and change the "scale" parameter (in very small steps, e.g. from 1.0 to 1.05).
To have a reliable ground truth, take a video in which you have single rotations (!) around each axis (e.g. 3x up - down - up - down - ... , 3x left - right - left - ..., 3x clockwise - counterclockwise - clockwise - ...)
Then repeat steps 4-6 and adjust the scale parameter slightly. Watch if your changes made it better or worse. Since this script caches the actual image analysis, this procedure can be done quite quickly in less than 15 minutes.
