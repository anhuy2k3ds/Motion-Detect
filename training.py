import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def process_video(video_path, label, img_size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    data = []
    labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame_resized = cv2.resize(frame, img_size)
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Chuẩn hóa giá trị pixel (normalize pixel values)
        gray_frame = gray_frame / 255.0
        
        data.append(gray_frame)
        labels.append(label)

    cap.release()
    return np.array(data), np.array(labels)

video_with_motion = "video/23796-337668530_small.mp4"
video_without_motion = "video\WIN_20241111_10_23_33_Pro.mp4"

# Chuẩn bị dữ liệu và nhãn (1: có chuyển động, 0: không có chuyển động)
data_motion, labels_motion = process_video(video_with_motion, label=1)
data_no_motion, labels_no_motion = process_video(video_without_motion, label=0)

# Gộp dữ liệu từ cả hai tập
data = np.concatenate((data_motion, data_no_motion), axis=0)
labels = np.concatenate((labels_motion, labels_no_motion), axis=0)

# Reshape lại dữ liệu để phù hợp với đầu vào của CNN (batch_size, height, width, channels)
data = data.reshape(-1, 64, 64, 1)

print("Dữ liệu đã xử lý: ", data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Tạo mô hình
model = build_model()

# Huấn luyện mô hình
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Lưu mô hình sau khi huấn luyện
model.save('motion_detection_model.h5')

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Độ chính xác của mô hình trên tập kiểm thử: {test_acc * 100:.2f}%")
