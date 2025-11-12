import os
import cv2
import time
import shutil
import numpy as np
import threading
from collections import deque
from utils.detectionSCRFD import SCRFD
from utils.embeddingsArcface import ArcFace
from milvusDB import MilvusDB
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.byte_tracker import BYTETracker # asumsi lo udah punya ini
from collections import OrderedDict

# =====================================================
# CONFIG
# =====================================================
DATABASE_PATH = "database"
PROFILE_DIR = "temp/faceProfile"
MATCH_THRESHOLD = 0.5
UNKNOWN_FOLDER = "unknown_faces"
SKIPFRAME = 2
trackThresh = 0.5
trackBuffer = 30
trackerFrameRate = 30
trackerMatchThresh = 0.8


# =====================================================
# CLASS
# =====================================================
class LimitedDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

# =====================================================
# INIT
# =====================================================
detector = SCRFD(model_path="models/buffalo_m/det_25g.onnx")
recognizer = ArcFace(model_path="models/buffalo_m/w600k_r50.onnx")

os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FOLDER, exist_ok=True)
os.makedirs(DATABASE_PATH, exist_ok=True)


# =====================================================
# FUNGSI PROCESS + BUILD DB
# =====================================================
def process_image(img_path, faceDB, move_after=True, padding=0):
    if not os.path.exists(img_path):
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Cannot read image: {img_path}")
        return

    name = os.path.splitext(os.path.basename(img_path))[0].split("-")[0]
    boxes, points_list = detector.detect(img)
    if boxes is None or len(boxes) == 0:
        print(f"[SKIP] No face detected in {img_path}")
        return

    box, kps = boxes[0], points_list[0]
    x1, y1, x2, y2, score = box.astype(np.int32)
    h, w = img.shape[:2]
    face_crop = img[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)].copy()

    os.makedirs(PROFILE_DIR, exist_ok=True)
    save_path = os.path.join(PROFILE_DIR, f"{name}.jpg")

    if not os.path.exists(save_path):
        shutil.move(img_path, save_path)
        print(f"[SAVE] Moved {img_path} → {save_path}")
    elif move_after:
        os.remove(img_path)

    emb = recognizer(img, kps)
    meta = {"name": name}
    faceDB.insert(emb, meta, MATCH_THRESHOLD)
    print(f"[DB] Inserted '{name}' to MilvusDB")


def build_face_database(folder_path, faceDB, padding=0):
    print(f"\n[INFO] Building Face Database from '{folder_path}'...")
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder {folder_path} not found!")
        return
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"[WARNING] No image files found in {folder_path}")
        return

    def sort_key(filename):
        name, _ = os.path.splitext(filename)
        parts = name.split("-")
        if len(parts) > 1 and parts[-1].isdigit():
            return (parts[0], int(parts[-1]))
        return (parts[0], 9999)

    image_files.sort(key=sort_key)
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        process_image(img_path, faceDB, move_after=True, padding=padding)


# =====================================================
# WATCHER
# =====================================================
class DatabaseWatcher(FileSystemEventHandler):
    def __init__(self, faceDB, padding=0):
        self.faceDB = faceDB
        self.padding = padding
        self.last_processed = 0
        self.min_interval = 1.0  # detik antar proses

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return

        now = time.time()
        if now - self.last_processed < self.min_interval:
            # Skip event kalau masih dalam interval < 1 detik
            return

        self.last_processed = now
        print(f"[WATCH] New image detected: {event.src_path}")

        # Jalanin di thread kecil supaya gak blocking
        threading.Thread(
            target=process_image,
            args=(event.src_path, self.faceDB),
            kwargs={'move_after': True, 'padding': self.padding},
            daemon=True
        ).start()


def start_watching(folder_path, faceDB, padding=0):
    observer = Observer()
    event_handler = DatabaseWatcher(faceDB, padding)
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f"[INFO] Watching folder: {folder_path}")
    observer.join()


# =====================================================
# STREAMING (MAIN LOOP)
# =====================================================
def run_stream(faceDB, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[ERROR] Camera not available")
        return

    tracker = BYTETracker(track_thresh=trackThresh, track_buffer=trackBuffer, frame_rate=trackerFrameRate)
    faceTracked = LimitedDict(10)
    faceKeypoints = LimitedDict(1000)
    frameCounting = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frameCounting % SKIPFRAME == 0:
            start_time = time.time()
            cleanFrame = frame.copy()
            boxes, points_list = detector.detect(frame)
            endDetectionTime = (time.time()-start_time)* 1000
            print(f"[INFO] detection time : {endDetectionTime:.2f} ms")

            if boxes is not None and len(boxes) > 0:
                detectedFaces = []
                for box, kps in zip(boxes, points_list):
                    x1, y1, x2, y2, score = box
                    detectedFaces.append([int(x1), int(y1), int(x2), int(y2), float(score)])
                    faceKeypoints[score] = kps
                detectedFaces = np.array(detectedFaces, dtype=np.float32)

                tracks = tracker.update(
                    detectedFaces,
                    [frame.shape[0], frame.shape[1]],
                    [frame.shape[0], frame.shape[1]],
                    mot20=False,
                    match_thresh=trackerMatchThresh
                )

                for face in tracks:
                    x1, y1, x2, y2 = map(int, face.tlbr)
                    faceId = face.track_id
                    kps = faceKeypoints[face.score]

                    if faceId not in faceTracked:
                        faceTracked[faceId] = deque(maxlen=10)

                    embeddingStartTime = time.time()
                    emb = recognizer(cleanFrame, kps)
                    faceTracked[faceId].append(emb)
                    embeddingEndTime = (time.time()-embeddingStartTime)* 1000
                    print(f"[INFO] embedding time : {embeddingEndTime:.2f} ms")

                    if len(faceTracked[faceId]) >= 3:
                        meanEmbStartTime = time.time()
                        mean_emb = np.mean(faceTracked[faceId], axis=0)
                        meanEmbEndTime = (time.time() - meanEmbStartTime)* 1000
                        print(f"[INFO] mean embedding time : {meanEmbEndTime:.2f} ms")
                        searchStartTime = time.time()
                        results = faceDB.search(mean_emb, 1)
                        endSearchTime = (time.time() - searchStartTime)* 1000
                        print(f"[INFO] search face time in database : {endSearchTime:.2f} ms")
                        name = "unknown"
                        if len(results):
                            print(f"score {results[0]['score']}, name : {results[0]['name']}")
                            if results and results[0]["score"] >= MATCH_THRESHOLD:
                                name = results[0]["name"]

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            elapsed = (time.time() - start_time) * 1000
            fps = 1000 / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"{elapsed:.1f} ms ({fps:.1f} FPS)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frameCounting += 1
        if frameCounting >= 100:
            frameCounting = 0

    cap.release()
    cv2.destroyAllWindows()


# =====================================================
# MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    FaceDB = MilvusDB("face_demo.db", dim=512)
    build_face_database(DATABASE_PATH, FaceDB)

    # Jalankan watcher di thread lain
    watcher_thread = threading.Thread(target=start_watching, args=(DATABASE_PATH, FaceDB), daemon=True)
    watcher_thread.start()

    # Jalankan stream di thread utama
    run_stream(FaceDB, 0)
