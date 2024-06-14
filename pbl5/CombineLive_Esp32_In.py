import urllib

import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import requests
import tensorflow as tf
import pickle
import imutils

detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('../testTrainModelDlib/predictor.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

url_cam = 'http://192.168.43.189/cam-hi.jpg'


#
protoPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the liveness detection model and label encoder
model = tf.keras.models.load_model("liveness.h5")
le = pickle.loads(open("le.pickle", "rb").read())

# Minimum probability to filter weak detections
confidence_threshold = 0.7

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.frame_cnt = 0

        self.face_features_known_list = []
        self.face_name_known_list = []

        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        self.current_frame_face_X_e_distance_list = []

        self.current_frame_face_position_list = []
        self.current_frame_face_feature_list = []

        self.last_current_frame_centroid_e_distance = 0

        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        self.face_id_list_known_list = []
        self.url_cam = url_cam

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                employee = {}
                self.face_name_known_list.append(csv_rd.iloc[i][0].split('_')[0])
                employee['id'] = csv_rd.iloc[i][0].split('_')[0]
                employee['name'] = csv_rd.iloc[i][0].split('_')[1]
                self.face_id_list_known_list.append(employee)
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []

            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    def draw_note(self, img_rd):
        # 添加说明 / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    def clear_csv(self):
        columns = ["Status", "Id", "Time", "PersonName", "StatusRecognized", "Img_Path"]
        pd.DataFrame(columns=columns).to_csv('dataRecognizeIn.csv', mode='w', header=True, index=False)
    def save_to_csv(self, data):
        self.clear_csv()
        columns = ["Status", "Id", "Time", "PersonName", "StatusRecognized", "Img_Path"]
        new_data = pd.DataFrame([data])
        if os.path.exists('dataRecognizeIn.csv'):
            new_data.to_csv('dataRecognizeIn.csv', mode='a', header=False, index=False)
        else:
            new_data.to_csv('dataRecognizeIn.csv', mode='w', header=columns, index=False)

    def check_liveness(self, frame):
        # Resize the frame and grab its dimensions
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # Convert the frame to a blob and pass it through the network
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        results = []
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than the threshold
            if confidence > confidence_threshold:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box dimensions are within the frame dimensions
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                if startX < endX and startY < endY:
                    # Extract the face ROI, resize it, and preprocess it for the liveness model
                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = tf.keras.utils.img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    # Pass the face through the liveness detection model to predict "real" or "fake"
                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    # Return the label (either "real" or "fake") and the bounding box coordinates
                    results.append(label)

        # If no faces are found, return "unknown" label and None for bounding box
        return results
    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")

                img_resp = urllib.request.urlopen(self.url_cam)
                img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, -1)
                gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_rd = frame

                kk = cv2.waitKey(1)

                results = self.check_liveness(img_rd)
                if all(label == 'real' for label in results):
                    faces = detector(gray_img, 0)
                    self.last_frame_face_cnt = self.current_frame_face_cnt
                    self.current_frame_face_cnt = len(faces)

                    self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                    self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                    self.current_frame_face_centroid_list = []

                    if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                            self.reclassify_interval_cnt != self.reclassify_interval):
                        logging.debug("scene 1: 当前帧和上一帧相比没有发生人脸数变化 / No face cnt changes in this frame!!!")

                        self.current_frame_face_position_list = []

                        if "unknown" in self.current_frame_face_name_list:
                            logging.debug("  有未知人脸, 开始进行 reclassify_interval_cnt 计数")
                            self.reclassify_interval_cnt += 1

                        if self.current_frame_face_cnt != 0:
                            for k, d in enumerate(faces):
                                self.current_frame_face_position_list.append(tuple(
                                    [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                                self.current_frame_face_centroid_list.append(
                                    [int(faces[k].left() + faces[k].right()) / 2,
                                     int(faces[k].top() + faces[k].bottom()) / 2])

                                img_rd = cv2.rectangle(img_rd,
                                                       tuple([d.left(), d.top()]),
                                                       tuple([d.right(), d.bottom()]),
                                                       (255, 255, 255), 2)

                        if self.current_frame_face_cnt != 1:
                            self.centroid_tracker()

                        for i in range(self.current_frame_face_cnt):
                            name = "unknown"
                            for employee in self.face_id_list_known_list:
                                if employee['id'] == self.current_frame_face_name_list[i]:
                                    name = employee['name']
                            img_rd = cv2.putText(img_rd, name,
                                                 self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                                 cv2.LINE_AA)
                        self.draw_note(img_rd)

                    else:
                        logging.debug("scene 2: 当前帧和上一帧相比人脸数发生变化 / Faces cnt changes in this frame")
                        self.current_frame_face_position_list = []
                        self.current_frame_face_X_e_distance_list = []
                        self.current_frame_face_feature_list = []
                        self.reclassify_interval_cnt = 0

                        if self.current_frame_face_cnt == 0:
                            logging.debug("  scene 2.1 人脸消失, 当前帧中没有人脸 / No faces in this frame!!!")
                            self.current_frame_face_name_list = []
                        else:
                            logging.debug("  scene 2.2 出现人脸, 进行人脸识别 / Get faces in this frame and do face recognition")
                            self.current_frame_face_name_list = []
                            for i in range(len(faces)):
                                shape = predictor(img_rd, faces[i])
                                self.current_frame_face_feature_list.append(
                                    face_reco_model.compute_face_descriptor(img_rd, shape))
                                self.current_frame_face_name_list.append("unknown")


                            for k in range(len(faces)):
                                logging.debug("  For face %d in current frame:", k + 1)
                                self.current_frame_face_centroid_list.append(
                                    [int(faces[k].left() + faces[k].right()) / 2,
                                     int(faces[k].top() + faces[k].bottom()) / 2])

                                self.current_frame_face_X_e_distance_list = []

                                self.current_frame_face_position_list.append(tuple(
                                    [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                                for i in range(len(self.face_features_known_list)):
                                    if str(self.face_features_known_list[i][0]) != '0.0':
                                        e_distance_tmp = self.return_euclidean_distance(
                                            self.current_frame_face_feature_list[k],
                                            self.face_features_known_list[i])
                                        logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                        self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                    else:
                                        self.current_frame_face_X_e_distance_list.append(999999999)

                                similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                    min(self.current_frame_face_X_e_distance_list))

                                if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                    self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                    logging.debug("  Face recognition result: %s",
                                                  self.face_name_known_list[similar_person_num])
                                else:
                                    logging.debug("  Face recognition result: unknown person")

                            print(self.current_frame_face_name_list)
                            self.draw_note(img_rd)

                            for id in self.current_frame_face_name_list:
                                if id != 'unknown' and len(self.current_frame_face_name_list) > 0:
                                    for i in range(self.current_frame_face_cnt):
                                        name = "unknown"
                                        for employee in self.face_id_list_known_list:
                                            if employee['id'] == self.current_frame_face_name_list[i]:
                                                name = employee['name']
                                    time_saved = int(time.time())
                                    img_path = f"E:/Pbl5/Client/PBL5_INSOMNIA/Web/Img/{id}_{time_saved}.jpg"
                                    cv2.imwrite(img_path, img_rd)
                                    data = {"Status": True, "Id": id, "Time": time_saved, "PersonName": name, "StatusRecognized": True, "Img_Path": time_saved}
                                    self.save_to_csv(data)

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("cameraIn", 1)
                cv2.imshow("cameraIn", img_rd)

                logging.debug("Frame ends\n\n")



    def run(self):
        cap = cv2.VideoCapture(0)
        time.sleep(2.0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():

    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()

