import os
from imutils import paths
import matplotlib.pyplot as plt

# Thay đổi đường dẫn dataset của bạn tại đây
dataset_path = "../dataset"

# Lấy danh sách đường dẫn tới các hình ảnh trong dataset
imagePaths = list(paths.list_images(dataset_path))

# Khởi tạo biến đếm số lượng ảnh real và fake
num_real_images = 0
num_fake_images = 0

# Lặp qua từng đường dẫn ảnh và đếm số lượng real và fake
for imagePath in imagePaths:
    # Extract class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # Increment corresponding counter based on label
    if label == "real":
        num_real_images += 1
    elif label == "fake":
        num_fake_images += 1

# In ra số lượng ảnh real và fake
print(f"Number of real images: {num_real_images}")
print(f"Number of fake images: {num_fake_images}")

# Vẽ biểu đồ cột
labels = ['Real', 'Fake']
counts = [num_real_images, num_fake_images]

plt.bar(labels, counts, color=['blue', 'red'])

# Ghi số lượng ở phía trên mỗi cột
for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=12)


plt.xlabel('Kiểu ảnh')
plt.ylabel('Số lợng ảnh')
plt.title('Số lượng ảnh real và fake của tập dữ liệu')
plt.show()
