import cv2
import face_recognition
import os
import csv
import time
from datetime import datetime
import numpy as np

# ---------- CONFIG ----------
DATASET_DIR = os.path.abspath("dataset")
ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

ATTENDANCE_FILE = os.path.join(
    ATTENDANCE_DIR,
    f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

known_encodings = []
known_names = []

print("🔄 Loading students...")

# ---------- LOAD DATASET ----------
for student in os.listdir(DATASET_DIR):
    student_path = os.path.join(DATASET_DIR, student)

    if not os.path.isdir(student_path):
        continue

    valid = False
    for img in os.listdir(student_path):
        if img.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(student_path, img)
            try:
                image = face_recognition.load_image_file(img_path)
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                encodings = face_recognition.face_encodings(image, num_jitters=2)

                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(student)
                    print("✅ Loaded:", student)
                    valid = True
                    break
                else:
                    print("❌ No face in:", img_path)
            except Exception as e:
                print("❌ Error reading:", img_path, e)

    if not valid:
        print("⚠️ No valid image for student:", student)

if not known_names:
    print("❌ NO STUDENTS LOADED → CHECK DATASET FOLDER")
    exit()

print("👥 Students:", known_names)

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera error")
    exit()

print("🎥 Camera started (10 seconds)...")

present = set()
start = time.time()

while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(
        rgb, number_of_times_to_upsample=2, model="hog"
    )
    encodings = face_recognition.face_encodings(rgb, locations)

    for enc, (top, right, bottom, left) in zip(encodings, locations):
        distances = face_recognition.face_distance(known_encodings, enc)
        best_match = np.argmin(distances)

        if distances[best_match] < 0.6:
            name = known_names[best_match]
            present.add(name)
        else:
            name = "Unknown"

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- SAVE ATTENDANCE ----------
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(ATTENDANCE_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Status", "Time"])

    for name in set(known_names):
        status = "Present" if name in present else "Absent"
        writer.writerow([name, status, now])

print("\n📋 Attendance Summary:")
for name in set(known_names):
    print(name, "→", "Present" if name in present else "Absent")

print("\n✅ Saved to:", ATTENDANCE_FILE)
