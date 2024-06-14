# Import các thư viện cần thiết
import multiprocessing
import argparse
import dlib
import matplotlib.pyplot as plt

# Giả sử rằng đường dẫn đến file XML và mô hình đã được định nghĩa
training_xml_path = "ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"
testing_xml_path = "ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml"
model_output_path = "calibrate.dat"

# Lưu trữ kết quả độ chính xác
errors = []

# Duyệt qua các giá trị của tree_depth từ 4 đến 10
for tree_depth in range(4, 11):
    print(f"[INFO] Training with tree_depth={tree_depth}...")

    # Cấu hình các tham số của dlib's shape predictor
    options = dlib.shape_predictor_training_options()
    options.tree_depth = tree_depth
    options.nu = 0.1
    options.cascade_depth = 15
    options.feature_pool_size = 400
    options.num_test_splits = 50
    options.oversampling_amount = 5
    options.oversampling_translation_jitter = 0.1
    options.be_verbose = True
    options.num_threads = multiprocessing.cpu_count()

    # Huấn luyện mô hình
    dlib.train_shape_predictor(training_xml_path, model_output_path, options)

    # Đánh giá mô hình và lưu giá trị lỗi
    error = dlib.test_shape_predictor(testing_xml_path, model_output_path)
    errors.append(error)
    print(f"[INFO] tree_depth={tree_depth}, error={error}")

# Vẽ biểu đồ độ chính xác
plt.figure()
plt.plot(range(4, 11), errors, marker='o')
plt.title("Error vs. Tree Depth")
plt.xlabel("Tree Depth")
plt.ylabel("Error")
plt.grid(True)
plt.show()