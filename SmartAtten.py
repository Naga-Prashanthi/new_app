import cv2
import os

# ---------- CONFIG ----------
DATASET_DIR = r"C:\Users\Dell\OneDrive\Desktop\newapp\dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------- INPUT ----------
name = input("Enter student name: ").strip().lower()
if not name:
    print("❌ Name empty")
    exit()

student_dir = os.path.join(DATASET_DIR, name)
os.makedirs(student_dir, exist_ok=True)

# ---------- FACE DETECTOR ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

count = 0
print("Press S to save | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not read")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1) & 0xFF

    # -------- SAVE FULL FRAME --------
    if key == ord('s'):
        if len(faces) == 0:
            print("⚠️ No face detected, not saving")
        else:
            count += 1
            img_path = os.path.join(student_dir, f"{count}.jpg")
            saved = cv2.imwrite(img_path, frame)
            print("✅ Saved:", saved, "→", img_path)

    if key == ord('q') or count >= 5:
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Registration done")
