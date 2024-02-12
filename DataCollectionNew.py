import os
import cv2

DATA_DIR = './data/0'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 200

# Open a VideoCapture object
cap = cv2.VideoCapture(0)

print('Hold "s" to capture and save an image.')
print('Press "q" to quit.')

saving = False
counter = 0
class_index = 0

while counter < dataset_size:
    ret, frame = cap.read()
    cv2.putText(frame, 'Hold "s" to capture and save, "q" to quit', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        saving = True
        counter += 1

    elif key == ord('q'):
        break

    if saving:
        save_path = os.path.join(DATA_DIR, str(class_index), f'{counter}.jpg')
        cv2.imwrite(save_path, frame)
        print(f"Image saved to {save_path}")
        saving = False

cap.release()
cv2.destroyAllWindows()
