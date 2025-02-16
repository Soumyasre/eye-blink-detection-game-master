import cv2
import time

# Load Haar cascade classifiers
face_cascade = cv2.CascadeClassifier(r"C:\globs\3-1 globs\eyeblinkdetection\Eye-blink-detection-game-master\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\globs\3-1 globs\eyeblinkdetection\Eye-blink-detection-game-master\haarcascade_eye_tree_eyeglasses.xml")

# Initialize variables
cap = cv2.VideoCapture(0)
blink_count = 0
blink_detected = False
start_time = time.time()  # Start timer
alert_shown = False  # To show alert message only once

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))
        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(face_region, 1.3, 5, minSize=(50, 50))
            
            if len(eyes) < 2:  # Eyes closed
                if not blink_detected:
                    blink_count += 1  # Count the blink only once per closure
                    blink_detected = True

                    # Show alert immediately if 10 blinks are detected
                    if blink_count >= 10 and not alert_shown:
                        print("ALERT: Frequent Blinking Detected!")
                        alert_shown = True

            else:  # Eyes open
                blink_detected = False

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Display blink count and timer on the screen
        cv2.putText(frame, f"Blinks: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Time: {int(elapsed_time)}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if alert_shown:
            cv2.putText(frame, "ALERT: Frequent Blinking Detected!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Eye Blink Detection', frame)

        # Exit if 'q' is pressed or if elapsed time reaches 15 seconds
        if cv2.waitKey(1) & 0xFF == ord('q') or elapsed_time >= 15:
            break

    print(f"Total Blinks: {blink_count}")
    if blink_count >= 10:
        print("Frequent blinking detected!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
