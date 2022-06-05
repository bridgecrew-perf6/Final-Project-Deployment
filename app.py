#!/usr/bin/env python
# coding: utf-8

from flask import Flask,render_template, Response,redirect,url_for
import cv2
import mediapipe as mp
import numpy as np
from requests import request

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
cap= cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mpPose= mp_pose.Pose()

def putText(img,text,loc,font_type,font_scale,font_color,dunno,dunno2):
                    cv2.putText(img,text,loc,font_type,font_scale,font_color,dunno,dunno2)

def situpFrames():
    stage = 'down'
    counter = 0
    label = None
    acceptable_position_error = 5
    x=0
    x=0

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
                
                # Get joints coordinates

                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                
                # Calculate angle
                # pilih pilih joints nya yang bener, pokoknya harus ngebentuk angle
                
                hip_angle = calculate_angle(shoulder, hip, knee)
                knee_angle = calculate_angle(hip, knee, ankle)

                
                condition_1 = hip_angle > 100
                condition_2 = hip_angle > 105
                condition_3 = hip_angle < (60 + acceptable_position_error)
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
                cv2.rectangle(image, (0,0), (350,73), (50, 168, 59), -1)
                
                #Curl counter logic
                if condition_1  :
                    stage = "down"
                    x=0
                    
                    if condition_2 :
                        label='Incorrect Rep'

                    
                if condition_3 :
                    
                    stage="up"
                    label='Correct Rep'
                    if x==0:
                            
                        counter+=1
                        print(counter)
                        print(label)
                    x=1



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
                
                # Condition data
                cv2.putText(image, 'LABEL', (270,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, label, 
                            (265,60), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2, cv2.LINE_AA)
                
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
                
                # Get joints coordinates

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
                

                


                condition_1 = elbow_angle > 100
                condition_2 = hip_angle < 150 or knee_angle<150
                condition_3 = elbow_angle > 100 and elbow_angle < 140
                condition_4 = elbow_angle<95
                condition_5= elbow_angle < 95 and hip_angle>150 and knee_angle>150 


                # Render angles

                cv2.putText(image, str(hip_angle), 
                            tuple(np.multiply(hip, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(knee_angle), 
                            tuple(np.multiply(knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                cv2.putText(image, str(elbow_angle), 
                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Setup status box
                cv2.rectangle(image, (0,0), (350,73), (50, 168, 59), -1)
                
                # Curl counter logic
                if condition_1:
                    stage = "up"
                    x=0
                    if condition_2:
                        label='Incorrect Pos'
                    elif condition_3:
                        label='Incorrect Rep'

                if condition_4:
                    stage='down'    

                if condition_5:
                    
                    
                    label='Correct Rep'
                    if x==0:
                            
                        counter+=1
                        print(counter)
                        print(label)
                    x=1

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
                
                # Condition data
                cv2.putText(image, 'LABEL', (270,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, label, 
                            (265,60), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2, cv2.LINE_AA)
                
                

                
                
                
                landmarks_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                
            except:
                pass

            # Render detections
            # drawLandmarks=mp_drawing.draw_landmarks()
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

@app.route('/pushup')
def index1():
    return render_template('pushup.html')

@app.route('/situp')
def index2():
    return render_template('situp.html')


@app.route('/video_feed')
def video_feed():
    return Response(situpFrames(),mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/pushup_feed')
def pushup_feed():
    return Response(pushupFrames(),mimetype='multipart/x-mixed-replace;boundary=frame')





# @app.route('/switch/',methods=('GET','POST'))
# def switch():
#     if request.method='POST':
#         return render_template('')
    
if __name__=='__main__':
    app.run(port=5000,debug=True)