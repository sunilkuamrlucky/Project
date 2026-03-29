import cv2
import numpy as np
import pickle
import os

GESTURES_FILE = 'gestures.pkl'

def load_gestures():
    if os.path.exists(GESTURES_FILE):
        with open(GESTURES_FILE, 'rb') as file:
            return pickle.load(file)
    return {}

def save_gestures(gestures):
    with open(GESTURES_FILE, 'wb') as file:
        pickle.dump(gestures, file)

def create_gesture(gesture_name):
    gestures = load_gestures()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 's' to save an image, 'q' to quit.")
    
    images = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Frame', gray)
        
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            images.append(gray)
            print(f"Image {len(images)} saved.")
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    gestures[gesture_name] = images
    save_gestures(gestures)
    print(f"Gesture '{gesture_name}' created with {len(images)} images.")

def display_gestures():
    gestures = load_gestures()
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

def main():
    while True:
        print("\nGesture Recognition System")
        print("1. Create Gesture")
        print("2. Display Gestures")
        print("3. Delete Gesture")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            gesture_name = input("Enter gesture name: ")
            create_gesture(gesture_name)
        elif choice == '2':
            display_gestures()
        elif choice == '3':
            gesture_name = input("Enter gesture name to delete: ")
            delete_gesture(gesture_name)
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
