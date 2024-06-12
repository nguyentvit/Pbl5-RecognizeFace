import os
import cv2
import xml.etree.ElementTree as ET

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def extract_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    boxes_with_parts = []

    for (x, y, w, h) in faces:
        box_info = {'box': (x, y, w, h), 'parts': []}
        boxes_with_parts.append(box_info)

    return boxes_with_parts


# Đường dẫn đến thư mục chứa dữ liệu LFW
lfw_dir = "lfw"

# Đường dẫn đến thư mục chứa tệp XML đầu ra
output_dir = "xmldata"

# Tạo thư mục đầu ra nếu nó chưa tồn tại
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Tạo phần tử root cho tệp XML
xml_root = ET.Element("images")

# Duyệt qua các thư mục và tệp trong tập dữ liệu LFW
for root, dirs, files in os.walk(lfw_dir):
    for file in files:
        # Xác định đường dẫn đầy đủ đến hình ảnh
        image_path = os.path.join(root, file)

        # Đọc hình ảnh
        image = cv2.imread(image_path)

        # Trích xuất các hộp khuôn mặt và các phần trong mỗi hộp
        boxes_with_parts = extract_boxes(image)

        # Thêm thông tin về hình ảnh vào tệp XML
        image_element = ET.SubElement(xml_root, "image")
        image_element.set("file", image_path)

        # Lưu thông tin về các hộp khuôn mặt và các phần vào tệp XML
        for box_info in boxes_with_parts:
            box_element = ET.SubElement(image_element, "box")
            box_element.set("top", str(box_info['box'][1]))
            box_element.set("left", str(box_info['box'][0]))
            box_element.set("width", str(box_info['box'][2]))
            box_element.set("height", str(box_info['box'][3]))

            # Thêm thông tin về các phần vào trong mỗi hộp
            for part in box_info['parts']:
                part_element = ET.SubElement(box_element, "part")
                part_element.set("x", str(part[0]))
                part_element.set("y", str(part[1]))

# Lưu tệp XML
xml_tree = ET.ElementTree(xml_root)
xml_output_path = os.path.join(output_dir, "images.xml")
xml_tree.write(xml_output_path)
