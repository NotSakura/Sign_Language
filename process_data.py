import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# extract the landmark position for each of the dataset images and store them in file to help train classifier (??)

# to detect the landmarks you need these:
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3)

DATA_DIR = './data/asl_alphabet_train'

data = []
labels = []
for dir_ in os.listdr(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.inread(os.path.join(DATA_DIR, dir_, img_path))
        # covert to rgb for reading in bgr
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:
                # to test if the thing wors and has the landmarks
                # mp_drawing.draw_landmarks(
                #     img_rgb, 
                #     hand_landmarks, 
                #     mp_hands.HAND_CONNECTIONS, 
                #     mp_drawing_styles.get_default_hand_landmarks_style(), 
                #     mp_drawing_styles.get_default_hand_connections_style(),
                # )  ##iterates through result(pics of hands) to draw the landmark lines and the connections

                for i in range(len(hand_landmarks.landmark)):
                    # print(hand_landmarks.landmark[i]) should print the x, y, z coordinate for each landmark
                    x, y = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            
            data.append(data_aux)
            labels.append(dir_) # the category (type of hand) of the dataset (account for this when you add the data from kaggle)

#         plt.figure()
#         plt.imshow(img_rgb)
    
# plt.show() # this is just to see if the pictures got converted and stored

f = open('data.pickle', 'wb') # save the info
pickle.dump({'data':data, 'labels': labels}, f)
f.close()


# this si the data we sue to train classifier