import pickle
import cv2
import mediapipe as mp
import numpy as np
import arabic_reshaper
import bidi.algorithm
from PIL import Image, ImageDraw, ImageFont

# Load the custom font
font_path = "arial.ttf"
font_size = 48  # Adjust the font size as needed

model_dict = pickle.load(open('./model.p.', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8)

labels_dict = {0: "مرحبا", 1: "لا", 2: "اتفق", 3: "رهيب", 4: "عمل جيد"}

while True:
    data_aux = []
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_word = labels_dict[int(prediction[0])]

        # Reshape the Arabic label using arabic_reshaper
        reshaped_label = arabic_reshaper.reshape(predicted_word)

        # Use the 'bidi' library to make the text appear RTL
        bidi_label = bidi.algorithm.get_display(reshaped_label)

        # Calculate the position to display the text on the frame
        x_text = 10
        y_text = 40  # Adjust the vertical position as needed

        # Create a PIL Image from the OpenCV frame
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Check if a valid hand gesture is detected (you can adjust the confidence threshold as needed)
        if result.multi_handedness[0].classification[0].index == 0:
            draw = ImageDraw.Draw(pil_image)
            font = ImageFont.truetype(font_path, font_size)

            # Get the text size using the textbbox method
            text_bbox = draw.textbbox((x_text, y_text), bidi_label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Render the reshaped Arabic label on the image using PIL
            draw.text((x_text, y_text), bidi_label, font=font, fill=(0, 0, 0))

        # Convert the PIL Image back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
