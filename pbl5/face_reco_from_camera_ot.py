import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import requests
import time

detector = dlib.get_frontal_face_detector()

# predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('../testTrainModelDlib/predictor.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

url_cam = 'http://172.20.10.3/cam-hi.jpg'
url_server = 'http://172.20.10.4:5126/api/admin/Attendances'


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

        self.url = url_server
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
        columns = ["Status", "id", "Time", "personName"]
        pd.DataFrame(columns=columns).to_csv('dataRecognizeIn.csv', mode='w', header=True, index=False)
    def save_to_csv(self, data):
        self.clear_csv()
        columns = ["Status", "id", "Time", "personName"]
        new_data = pd.DataFrame([data])
        if os.path.exists('dataRecognizeIn.csv'):
            new_data.to_csv('dataRecognizeIn.csv', mode='a', header=False, index=False)
        else:
            new_data.to_csv('dataRecognizeIn.csv', mode='w', header=columns, index=False)

    def process(self, stream):
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                faces = detector(img_rd, 0)

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

                        # self.url là url của server nha lê đần
                        # for id in self.current_frame_face_name_list:
                        #     if id != 'unknown' and len(self.current_frame_face_name_list) > 0:
                        #         time_saved = int(time.time())
                        #         data = {"status": False, "userId": id, "pathImg": time_saved}
                        #         image_path = f"E:/Pbl5/Client/PBL5_INSOMNIA/Web/Img/{id}_{time_saved}.jpg"
                        #         cv2.imwrite(image_path, img_rd)
                        #         response = requests.post(self.url, json=data)
                        #         if response.status_code == 201:
                        #             print("success")
                        #         else:
                        #             print("error")
                        for id in self.current_frame_face_name_list:
                            if id != 'unknown' and len(self.current_frame_face_name_list) > 0:
                                time_saved = int(time.time())
                                data = {"Status": False, "id": id, "Time": time_saved, "personName": "TranVanNguyen"}
                                self.save_to_csv(data)

                        # server
                        # @app.route('/')
                        # def home():
                        #     for id in self.current_frame_face_name_list:
                        #         if id != 'unknown' and len(self.current_frame_face_name_list) > 0:
                        #             return "Nhan dien thanh cong"
                        #         else:
                        #             return "Nhan dien khong thanh cong"

                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(0)

        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():

    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()

