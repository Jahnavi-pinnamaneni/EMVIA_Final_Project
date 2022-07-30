import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

image_width=1280
image_height=720
flag=1

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          print(
                  f'Index finger tip coordinates: (',
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width})'
                  f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height})'
              )
          Index_fingerX=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
          Thumb_fingerX=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width

          #print('fingertipX: '+str(fingertipX))
          #print(mp_hands.HandLandmark.INDEX_FINGER_TIP)
          rows, cols, _channels = map(int, image.shape)
          
          if (abs(Index_fingerX - Thumb_fingerX)) > 80 :
            print("Zoom Out")
            if flag == 0:
              image = cv2.pyrUp(image, dstsize=(2*cols, 2*rows))
              flag = 1
          else:
            print("Zoom In")
            if flag == 1:
              image = cv2.pyrDown(image, dstsize=(cols // 2, rows // 2))
              flag = 0
          mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
          cv2.imshow('MediaPipe Hands', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
