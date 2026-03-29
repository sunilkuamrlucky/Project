import cv2
import numpy as np
import pickle
import os

GESTURES_FILE = 'gestures.pkl'

def load_gestures():
    if os.path.exists(GESTURES_FILE):
        try:
            with open(GESTURES_FILE, 'rb') as file:
                return pickle.load(file)
        except (pickle.UnpicklingError, EOFError) as e:
            print("Error loading gestures file:", e)
            return {}
    return {}

def save_gestures(gestures):
    with open(GESTURES_FILE, 'wb') as file:
        pickle.dump(gestures, file)

def create_gesture(gesture_name):
    if not gesture_name.strip():
        print("Invalid gesture name.")
        return

    gestures = load_gestures()
    if gesture_name in gestures:
        print(f"Gesture {gesture_name} already exists.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 's' to save an image, 'q' to quit.")

    images = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Frame', gray)

            key = cv2.waitKey(1)
            if key == ord('s'):
                images.append(gray)
                print(f"Image {len(images)} saved.")
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if images:
        gestures[gesture_name] = images
        save_gestures(gestures)
        print(f"Gesture '{gesture_name}' created with {len(images)} images.")
    else:
        print("No images saved. Gesture not created.")

def display_gestures():
    gestures = load_gestures()
    if not gestures:
        print("No gestures found.")
        return

    for name, images in gestures.items():
        print(f"Gesture: {name} ({len(images)} images)")
        for i, img in enumerate(images):
            cv2.imshow(f"{name} - Image {i+1}", img)
            cv2.waitKey(0)
            cv2.destroyWindow(f"{name} - Image {i+1}")

def delete_gesture(gesture_name):
    gestures = load_gestures()

    if gesture_name in gestures:
        del gestures[gesture_name]
        save_gestures(gestures)
        print(f"Gesture '{gesture_name}' deleted.")
    else:
        print(f"Gesture '{gesture_name}' not found.")

def recognize_gesture(frame, gestures):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    best_match_name = None
    best_match_score = float('inf')

    for gesture_name, images in gestures.items():
        for template in images:
            if gray_frame.shape != template.shape:
                continue
            mse = np.mean((gray_frame.astype("float") - template.astype("float")) ** 2)
            if mse < best_match_score:
                best_match_score = mse
                best_match_name = gesture_name

    return best_match_name

def main():
    gestures = load_gestures()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            recognized_gesture = recognize_gesture(frame, gestures)
            if recognized_gesture:
                cv2.putText(frame, f"Gesture: {recognized_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Gesture Recognition', frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\nGesture Recognition System")
        print("1. Create Gesture")
        print("2. Display Gestures")
        print("3. Delete Gesture")
        print("4. Start Real-Time Recognition")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            gesture_name = input("Enter gesture name: ").strip()
            create_gesture(gesture_name)
        elif choice == '2':
            display_gestures()
        elif choice == '3':
            gesture_name = input("Enter gesture name to delete: ").strip()
            delete_gesture(gesture_name)
        elif choice == '4':
            main()
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")
