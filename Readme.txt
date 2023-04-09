------- Khalid -Yusuf-Liman code adapted from https://github.com/klimanyusuf/face_docker/tree/main/Daniel-Ogunlolu --

INFO:
Date: 7, April 2023
Programming Language: Python 3
Main Script: extractor.Python
Docker Base Image: orgoro/dlib-opencv-python

ALGORITHM:
1. Program starts by Downloading the given video from youtube link: 
https://www.youtube.com/watch?v=JriaiYZZhbY&t=4s
2. Saves video inside directory (Storage/) with extension .mp4
3. Run HOG face detection on each frame, all detections are saved in data/faces directory
4. Finally, a Clustering process is performed on all faces to filter  unique faces
5. 5 faces were skip to make it faster and a cluster of 5 was choosen as well
5. Unique faces are saved in data/unique_faces directory
6. When Program is done, enter q to end and exit the dockeer container

NOTE:
After building and running the docker image/container, you can copy directory containing 
faces extracted (data) using the following command:

docker cp <container id>:/main/data/ <host/path/to copy to/>

To view container id, use the following command:
docker ps

replace <host/path/to copy to/> with path where you want the data directory to be copied to.

eg: 
docker cp ad9f3d69bde2:/main/data/ /home/amazing/Desktop

RESULT:
The faces appeared blurry (and very small). This could be as a result of the quality of the video which is low. T

Tested the program with a diffrent video which has a better quality (faces were more vivid and clear) 
and the program worked well.

 A folder containing the final result (unique-faces) is attached.