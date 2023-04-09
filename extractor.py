import os.path
from sklearn.cluster import KMeans
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from imutils import face_utils
import dlib
import os
import sys
import pip
import cv2
import face_recognition
import numpy as np
from pytube import YouTube


# ------- We would create a folder: Storage where the video would be stored and other subfolders namely faces and uinque-fgace
os.chdir("storage")
imdir = ('faces')
targetdir = "unique-faces"
link = 'https://www.youtube.com/watch?v=JriaiYZZhbY&t=4s'


# creating clusters for the new faces

number_clusters = 5  # number of cluster for unique face
frame_skip = 5  # skip every fifth frame (makes the detection process faster)

if not os.path.isdir(imdir):
    os.mkdir("faces")
if not os.path.isdir(targetdir):
    os.mkdir(targetdir)

# --------------Dowloading video from the link

# function to download youtube video
def downloadYouTube(url, path):
    yt = YouTube(url)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by(
        'resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(path)


print("\n \nContainer Started")
print(" >>> Downloading Video from youtube link: {}".format(link))
downloadYouTube(link, '.')
print(" >>> Download done ")

for file in os.listdir():
    if file.endswith("mp4"):
        video = file
print(" >>> Video name: {}".format(video))

# --------- Building the face detection model

face_detect = dlib.get_frontal_face_detector()
# uncomment this to use cnn face detection whiich is slower
# face_detect = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
frame_no = 0
video_capture = cv2.VideoCapture(video)


# face detection fuction
def detect_face(frame):
    frame_original = frame
    global frame_no

    if frame_no % frame_skip == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect
        rects = face_detect(gray, 1)

        detect = 1
        for (i, rect) in enumerate(rects):
            detect = 2
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # save face
            cv2.imwrite("faces/frame-{}-face-{}.jpg".format(str(frame_no),
                        str(i)), cv2.resize(frame_original[y:y+h, x:x+w], (200, 200)))

    # return frame with detection
    return frame, detect


print(" >>> HOG is detecting Faces in all frames (loading)..........")

while True:
    # Capture frame
    ret, frame = video_capture.read()

    if ret == 0:
        break

    # Display the frame with detection
    try:
        show, detect = detect_face(frame)
        # cv2.imshow('Video', show)
    except Exception as e:
        # print(" >>> can not display video due to this error: {}".format(e))
        pass
    frame_no = frame_no + 1

    # Quit video by typing Q
    if cv2.waitKey(detect) & 0xFF == ord('q'):
        break

print(" >>> Reach end of Video or Can't receive frame")
print(" >>> Done, existing video window  and saving detected face")

video_capture.release()
# cv2.destroyAllWindows()

# clustering the faces detected
image.LOAD_TRUNCATED_IMAGES = True
model = VGG16(weights='imagenet', include_top=False)

# Variable
images = []

print("\n \n -------------------------------------------------------------------------------------------------------------------------------------------")
print(" >>> Performing Face Clustering to detect unique faces in faces saved in faces directory")
# Loop over files and get features
filelist = os.listdir(imdir)
for i, m in enumerate(filelist):
    filelist[i] = imdir + "/" + m
    images.append(cv2.imread(filelist[i]))
featurelist = []
for i, imagepath in enumerate(filelist):
    print(" >>> Status: %s / %s" % (i, len(filelist)), end="\r")
    img = image.load_img(imagepath, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data))
    featurelist.append(features.flatten())

# Clustering
kmeans = KMeans(n_clusters=number_clusters,
                random_state=0).fit(np.array(featurelist))

print("\n")

for i, label in enumerate(kmeans.labels_):
    dir = targetdir + "/" + "face-" + str(label)
    name = targetdir + "/" + "face-" + str(label) + "/" + str(i) + ".bmp"
    try:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        cv2.imwrite(name, images[i])
    except Exception as e:
        pass


print(" >>> Done")
print(" >>> Unique Faces saved to Unique-face directory")
print(" >>> Press q to exit program")

while (input(" >>> ") != 'q'):
    pass

print("Exiting......")
