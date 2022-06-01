#!/usr/bin/env python
# coding: utf-8

from flask import Flask,render_template, Response
import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

app= Flask(__name__)
cap= cv2.VideoCapture("pup.mp4")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def genFrames():
    stage = None
    counter = 0
    label = None
    acceptable_position_error = 5

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        success,frame=cap.read()
        if not success:
            break
        else:
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                # tinggal ganti aja mau joint yang mana, liat gambar joints yang direcognise mediapipe di atas
                # copy paste satu line di sini, terus ganti aja misal LEFT_HIP jadi LEFT_WRIST
                # jangan lupa ganti nama variable
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                
                # Calculate angle
                # pilih pilih joints nya yang bener, pokoknya harus ngebentuk angle
                
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)

                # Visualise angle
                # copy paste aja sefunction nya, cuma perlu ganti angka angle nya dari variable di atas
                # sama joint mana yang jadi angle

                condition_1 = hip_angle > (90+22.5 + acceptable_position_error)
                condition_2 = hip_angle < 45 - acceptable_position_error
                condition_3 = knee_angle < 90 - acceptable_position_error
                condition_4 = knee_angle > 90 + acceptable_position_error

                if condition_1 or condition_2 or condition_3 or condition_4:
                    label = "Incorrect"
                else:
                    label = "correct"

                # Render angles
                cv2.putText(image, str(hip_angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(knee_angle), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (50, 168, 59), -1)
                
                # Curl counter logic
                if hip_angle > 100:
                    stage = "down"
                if hip_angle < 60 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)

                # Rep data
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, 'STAGE', (155,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (145,60), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.75, (255,255,255), 2, cv2.LINE_AA)
                
                landmarks_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  

            ret,buffer=cv2.imencode('.jpg',image)    
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def pushupFrames():
    stage = None
    counter = 0
    label = None
    acceptable_position_error = 5

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        success,frame=cap.read()
        if not success:
            break
        else:
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                # tinggal ganti aja mau joint yang mana, liat gambar joints yang direcognise mediapipe di atas
                # copy paste satu line di sini, terus ganti aja misal LEFT_HIP jadi LEFT_WRIST
                # jangan lupa ganti nama variable
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                # Calculate angle
                # pilih pilih joints nya yang bener, pokoknya harus ngebentuk angle
                
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)
                elbow_angle= calculate_angle(shoulder,elbow,wrist)
                

                # Visualise angle
                # copy paste aja sefunction nya, cuma perlu ganti angka angle nya dari variable di atas
                # sama joint mana yang jadi angle

                condition_1 = hip_angle > (90+22.5 + acceptable_position_error)
                condition_2 = hip_angle < 45 - acceptable_position_error
                condition_3 = knee_angle < 90 - acceptable_position_error
                condition_4 = knee_angle > 90 + acceptable_position_error

                if condition_1 or condition_2 or condition_3 or condition_4:
                    label = "Incorrect"
                else:
                    label = "correct"

                # Render angles
                cv2.putText(image, str(hip_angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(knee_angle), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Setup status box
                cv2.rectangle(image, (0,0), (225,73), (50, 168, 59), -1)
                
                # Curl counter logic
                if elbow_angle > 100:
                    stage = "up"
                if elbow_angle < 85 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)

                # Rep data
                cv2.putText(image, 'REPS', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), 
                            (30,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
                
                # Stage data
                cv2.putText(image, 'STAGE', (155,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (145,60), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.75, (255,255,255), 2, cv2.LINE_AA)
                
                landmarks_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )  

            ret,buffer=cv2.imencode('.jpg',image)    
            frame=buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(pushupFrames(),mimetype='multipart/x-mixed-replace;boundary=frame')

    
if __name__=='__main__':
    app.run(port=5000,debug=True)