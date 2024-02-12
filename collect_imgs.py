import os
import cv2

DATA_DIR = './vald'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

    counter = 0
    half_dataset = dataset_size // 2
    paused = False

    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break

        if key == ord('p') or (counter == half_dataset and not paused):
            paused = not paused
            while paused:
                cv2.putText(frame, 'Paused. Press "p" to resume or "q" to quit.', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                            (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(25) & 0xFF
                if key == ord('p'):
                    paused = not paused
                elif key == ord('q'):
                    break

        if not paused:
            cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
            counter += 1

cap.release()
cv2.destroyAllWindows()
