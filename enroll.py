"""
enroll.py — command-line tool to enroll a mock employee via webcam.

Usage:
    python enroll.py

Captures several frames, averages their embeddings into one, and saves
the employee to the database. Averaging over a few shots makes the
saved embedding more robust than relying on a single photo.
"""
import cv2
import numpy as np
from face_pipeline import FacePipeline
from db import init_db, add_employee

NUM_CAPTURES = 5


def main():
    init_db()
    pipeline = FacePipeline()

    name = input("Employee name: ").strip()
    department = input("Department: ").strip()
    shift_start = input("Shift start time (HH:MM, default 09:00): ").strip() or "09:00"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    embeddings = []
    print(f"\nLook at the camera. Press 'c' to capture ({NUM_CAPTURES} captures needed), 'q' to quit.")

    while len(embeddings) < NUM_CAPTURES:
        ok, frame = cap.read()
        if not ok:
            break

        faces = pipeline.detect_faces(frame)
        display = frame.copy()
        if faces is not None and len(faces) > 0:
            box = faces[0][:4].astype(int)
            cv2.rectangle(display, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

        cv2.putText(display, f"Captures: {len(embeddings)}/{NUM_CAPTURES}  (c=capture, q=quit)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll Employee", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if faces is None or len(faces) == 0:
                print("No face detected, try again.")
                continue
            emb = pipeline.get_embedding(frame, faces[0])
            embeddings.append(emb)
            print(f"Captured {len(embeddings)}/{NUM_CAPTURES}")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == 0:
        print("No captures taken, nothing saved.")
        return

    avg_embedding = np.mean(embeddings, axis=0)
    add_employee(name, department, avg_embedding, shift_start)
    print(f"\nEnrolled '{name}' ({department}, shift starts {shift_start}).")


if __name__ == "__main__":
    main()
