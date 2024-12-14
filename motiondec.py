import cv2
import numpy as np
import smtplib
import tensorflow as tf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from tkinter import *
from tkinter import filedialog
import threading 

#----------------- Email Notification Setup -----------------

def send_email_alert():
    sender_email = "tvu051344@gmail.com"
    receiver_email = "20212004@eaut.edu.vn"
    password = "cvxu ywpj wccm ugxx"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Motion Detected!"

    body = "Motion detected in the video feed!"
    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, password)
    text = msg.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()

# Model (Optional)
model = tf.keras.models.load_model('motion_detection_model.h5')
# Motion Detection 
def motion_detection(video_source, model):
    cap = cv2.VideoCapture(video_source) 
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    frame_counter = 0
    
    def process_prediction(roi):
        resized_roi = cv2.resize(roi, (64, 64))
        gray_roi = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
        normalized_roi = gray_roi / 255.0
        input_data = np.expand_dims(normalized_roi, axis=(0, -1))
        
        prediction = model.predict(input_data)
        if prediction > 0.5:  
            send_email_alert()

    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            roi = frame1[y:y+h, x:x+w]  
            if frame_counter % 5 == 0:
                threading.Thread(target=process_prediction, args=(roi,)).start()

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Motion Detection", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#  UI Setup 
def open_camera():
    motion_detection(0, model) 

def open_video_file():
    video_path = filedialog.askopenfilename()
    if video_path:
        motion_detection(video_path, model)

# Tkinter UI 
root = Tk()
root.title("Motion Detection System")

label = Label(root, text="Select input source", font=("Helvetica", 16), )
label.pack(pady=20)

camera_button = Button(root, text="Open Camera", command=open_camera, padx=20, pady=10)
camera_button.pack(pady=10)

video_button = Button(root, text="Open Video File", command=open_video_file, padx=20, pady=10)
video_button.pack(pady=10)

root.mainloop()