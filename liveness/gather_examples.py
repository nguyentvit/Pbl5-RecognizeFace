# USAGE
# python gather_examples.py --input videos --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input video_list.txt --output dataset/fake --detector face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from glob import glob

def process_video(video_path, output_dir, net, skip, confidence, read_before):
    # open a pointer to the video file stream and initialize the total
    # number of frames read and saved thus far
    vs = cv2.VideoCapture(video_path)
    read = read_before
    saved = 0

    # extract the video filename (without extension) to use in the saved images' names
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # loop over frames from the video file stream
    while True:
        # grab the frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # increment the total number of frames read thus far
        read += 1

        # check to see if we should process this frame
        if read % skip != 0:
            continue

        # grab the frame dimensions and construct a blob from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence_val = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence_val > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

                # write the frame to disk
                p = os.path.sep.join([output_dir, "{}_{}.png".format(video_filename, saved)])
                if face.size > 0:  # Ensure the face ROI is not empty
                    cv2.imwrite(p, face)
                    saved += 1
                    print("[INFO] saved {} to disk".format(p))

    # do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()
    return read

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video or directory containing videos")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Determine if input is a directory or a file
if os.path.isdir(args["input"]):
    video_paths = glob(os.path.join(args["input"], "*.*"))  # Match all files
    video_paths = [f for f in video_paths if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]  # Filter video files
else:
    with open(args["input"], 'r') as file:
        video_paths = [line.strip() for line in file]

# Create output directory if it does not exist
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

read_before = 0
# Process each video file
for video_path in video_paths:
    print(f"[INFO] processing video: {video_path}")
    read_before = process_video(video_path, args["output"], net, args["skip"], args["confidence"], read_before)
# USAGE
# python gather_examples.py --input videos --output dataset/real --detector face_detector --skip 1
# python gather_examples.py --input video_list.txt --output dataset/fake --detector face_detector --skip 4

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from glob import glob

def process_video(video_path, output_dir, net, skip, confidence, read_before):
    # open a pointer to the video file stream and initialize the total
    # number of frames read and saved thus far
    vs = cv2.VideoCapture(video_path)
    read = read_before
    saved = 0

    # extract the video filename (without extension) to use in the saved images' names
    video_filename = os.path.splitext(os.path.basename(video_path))[0]

    # loop over frames from the video file stream
    while True:
        # grab the frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        # increment the total number of frames read thus far
        read += 1

        # check to see if we should process this frame
        if read % skip != 0:
            continue

        # grab the frame dimensions and construct a blob from the frame
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # ensure at least one face was found
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence_val = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
            if confidence_val > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]

                # write the frame to disk
                p = os.path.sep.join([output_dir, "{}_{}.png".format(video_filename, saved)])
                if face.size > 0:  # Ensure the face ROI is not empty
                    cv2.imwrite(p, face)
                    saved += 1
                    print("[INFO] saved {} to disk".format(p))

    # do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()
    return read

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video or directory containing videos")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
                              "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Determine if input is a directory or a file
if os.path.isdir(args["input"]):
    video_paths = glob(os.path.join(args["input"], "*.*"))  # Match all files
    video_paths = [f for f in video_paths if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]  # Filter video files
else:
    with open(args["input"], 'r') as file:
        video_paths = [line.strip() for line in file]

# Create output directory if it does not exist
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

read_before = 0
# Process each video file
for video_path in video_paths:
    print(f"[INFO] processing video: {video_path}")
    read_before = process_video(video_path, args["output"], net, args["skip"], args["confidence"], read_before)
