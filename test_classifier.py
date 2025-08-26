import cv2
import mediapipe as mp
import pickle
import numpy as np


cap = cv2.VideoCapture(0)

# load model from before
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#media pipe functions from processing data file

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3)

labels_dict = {0:'A', 1: 'B', 2:'L'} # this needs to change as this labels is based on the dataset that the video has but we are importing from kaggle

while True:
    data_aux = []
    x_, y_ = [], []
    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    res = hands.process(frame_rgb)
    if res.multi_hand_landmarks:
        for hand_landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_drawing.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(), 
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in res.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                # print(hand_landmarks.landmark[i]) should print the x, y, z coordinate for each landmark
                x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        x1 = int(min(x_) * W) -10 # get the corners of the rectangle covering the hand
        y1 = int(min(y_) * H) -10

        x2 = int(max(x_) * W) -10
        y2 = int(max(y_) * H) -10

        prediction = model.predict([np.asarray(data_aux)])
        # prediction is a lits of one element so we must parse through it below
        predicted_char = labels_dict[int(prediction[0])]

        print(predicted_char)

    
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4) # last 2 is colour and thickness
        cv2.putText(frame, predicted_char, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    cv2.waitKey(25) # wait 25 seconds for each frame (reduce for real time)

cap.release()
cv2.destroyAllWindows()