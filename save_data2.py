import cv2
import mediapipe as mp
import csv
import os
import itertools

# --- Mediapipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([x, y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()

    # tọa độ tương đối
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    # vector 1 chiều
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # chuẩn hóa
    max_value = max(list(map(abs, temp_landmark_list))) if max(list(map(abs, temp_landmark_list))) != 0 else 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

def process_dataset(dataset_dir="D:/AIRC/Raw_Data2", 
                    csv_path="D:/AIRC/keypoint_classifier/keypoint2.csv"):
    # Tạo map folder -> số
    gesture_list = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    gesture_list = sorted(gesture_list)  # sắp xếp alphabet
    gesture_to_id = {gesture_name: idx for idx, gesture_name in enumerate(gesture_list)}
    print("📌 Label mapping:", gesture_to_id)

    with open(csv_path, 'w', newline="") as f:
        writer = csv.writer(f)

        for gesture_name in gesture_list:
            gesture_path = os.path.join(dataset_dir, gesture_name)
            print(f"👉 Đang xử lý class: {gesture_name}")

            for file_name in os.listdir(gesture_path):
                if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
                    continue

                file_path = os.path.join(gesture_path, file_name)
                image = cv2.imread(file_path)
                if image is None:
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmark_list = calc_landmark_list(image, hand_landmarks)
                        pre_processed = pre_process_landmark(landmark_list)

                        # Lưu: [label_id, landmarks...]
                        writer.writerow([gesture_to_id[gesture_name], *pre_processed])

    print(f"✅ Done! Dữ liệu đã lưu vào {csv_path}")

# --- Chạy ---
if __name__ == "__main__":
    process_dataset("D:/AIRC/Raw_Data2", "D:\AIRC\keypoint_classifier\keypoint2.csv")
