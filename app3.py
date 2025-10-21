import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"D:/AIRC/model.tflite")
interpreter.allocate_tensors()

# Lấy thông tin input/output tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","del","nothing","space"]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def pre_process_landmarks(landmarks):
    landmark_array = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    base_x, base_y = landmark_array[0]
    landmark_array -= [base_x, base_y]
    max_value = np.max(np.abs(landmark_array))
    if max_value > 0:
        landmark_array /= max_value
    return landmark_array.flatten()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không mở được camera")
    exit()

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        image.flags.writeable = True
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Chuẩn bị input cho model
                landmark_list = pre_process_landmarks(hand_landmarks.landmark)
                input_data = np.array([landmark_list], dtype=np.float32)

                # Chạy model TFLite
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                # Lấy kết quả
                class_id = np.argmax(output_data[0])
                class_label = labels[class_id]

                cv2.putText(image, class_label, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # cv2.imshow('MediaPipe Hands + TFLite Classifier', cv2.flip(image, 1))
        cv2.imshow('MediaPipe Hands + TFLite Classifier', image)

        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
