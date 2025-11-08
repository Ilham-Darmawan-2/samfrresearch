import cv2
import numpy as np
import time
import face_recognition
from utils.byte_tracker import BYTETracker

# ================= CONFIG =================
VIDEO_PATH = "rtsp://admin:GM_282802@192.168.10.240:554/media/video2"
FRAME_RESIZE = (1280, 720)
SKIP_FRAME = 3
MATCHTHRESH = 0.7

# ==========================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    tracker = BYTETracker(track_thresh=0.5, track_buffer=30)
    frameNumber = 1

    print("[INFO] Starting face detection with BYTETracker...")

    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to grab frame.")
            break

        if frameNumber % SKIP_FRAME == 0:
            start_time = time.time()

            # Ubah ke RGB untuk face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Deteksi wajah (format: top, right, bottom, left)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            dets = []
            for (top, right, bottom, left) in face_locations:
                # Konversi ke xyxy
                x1, y1, x2, y2 = left, top, right, bottom
                conf = 1.0  # face_recognition gak kasih confidence, jadi diset 1.0
                dets.append([x1, y1, x2, y2, conf])

            if len(dets) > 0:
                dets = np.array(dets)
                tracks = tracker.update(
                    dets,
                    [frame.shape[0], frame.shape[1]],
                    [frame.shape[0], frame.shape[1]],
                    mot20=False,
                    match_thresh=MATCHTHRESH
                )

                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.tlbr)
                    track_id = track.track_id
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"id {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # tampilkan FPS/inference
            inference_ms = (time.time() - start_time) * 1000
            cv2.putText(frame, f"Inference: {inference_ms:.1f} ms", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            resized = cv2.resize(frame, FRAME_RESIZE)
            cv2.imshow("BYTETrack Face Tracking", resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frameNumber += 1
        if frameNumber >= 100:
            frameNumber = 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
