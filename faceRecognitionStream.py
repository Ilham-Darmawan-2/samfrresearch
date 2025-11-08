import os
import cv2
import numpy as np
import time
from utils.detectionSCRFD import SCRFD
from utils.embeddingsArcface import ArcFace
from utils.frutility import draw_corners, draw_keypoints
from utils.byte_tracker import BYTETracker
from collections import OrderedDict, deque

DATABASE_PATH = "database"
UNKNOWN_FOLDER = "unknown"
FRAME_RESIZE = (1280, 720)

trackThresh = 0.3
trackBuffer = 30
trackerFrameRate = 40
trackerMatchThresh = 0.6
MATCH_THRESHOLD = 0.25
SKIPFRAME = 5
RECOOLDOWN = 3  # detik

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
    
# ================================
# Inisialisasi detector & recognizer
# ================================
detector = SCRFD(model_path="models/buffalo_m/det_25g.onnx")
recognizer = ArcFace(model_path="models/buffalo_m/w600k_r50.onnx")

# =====================================================
# FACE DATABASE
# =====================================================
def build_face_database(folder_path, padding=0):

    database = []
    print(f"\n[INFO] Building Face Database from '{folder_path}'...")

    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder {folder_path} not found!")
        return database

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Dapat bounding box & keypoints
        boxes, points_list = detector.detect(img)
        if boxes is not None and len(boxes) > 0:
            # Ambil wajah pertama
            box, kps = boxes[0], points_list[0]
            x1, y1, x2, y2, score = box.astype(np.int32)

            # =========================
            # Crop wajah + padding
            # =========================
            h, w = img.shape[:2]
            x1p = max(x1 - padding, 0)
            y1p = max(y1 - padding, 0)
            x2p = min(x2 + padding, w - 1)
            y2p = min(y2 + padding, h - 1)

            face_crop = img[y1p:y2p, x1p:x2p].copy()
            cv2.imwrite("crop.jpg", face_crop)

            emb = recognizer(img, kps)

            name = os.path.splitext(img_name)[0].split("-")[0]
            database.append({"name": name, "embedding": emb})
            print(f"[OK] Added '{img_name}' as '{name}'")

    print(f"[DONE] Total faces in DB: {len(database)}\n")
    return database

# =====================================================
# FACE COMPARISON
# =====================================================
def compare_embeddings(emb1, emb2):
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return sim

def get_last_unknown_number(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return 0

    files = [f for f in os.listdir(folder_path) if f.startswith("unknown-")]
    if not files:
        return 0
    numbers = []
    for f in files:
        try:
            num = int(f.split("-")[1].split(".")[0])
            numbers.append(num)
        except Exception:
            pass
    return max(numbers) if numbers else 0

cap = cv2.VideoCapture("rtsp://admin:GM_282802@192.168.10.236:554/Streaming/Channels/102")
FaceDB = build_face_database(DATABASE_PATH)
unknownNumber = get_last_unknown_number(UNKNOWN_FOLDER) + 1
tracker = BYTETracker(track_thresh=trackThresh, track_buffer=trackBuffer, frame_rate=trackerFrameRate)
faceTracked = LimitedDict(10)
faceKeypoints = LimitedDict(1000)
recognized_cache = {}
frameCounting = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ====================
    # Deteksi wajah
    # ====================
    if frameCounting % SKIPFRAME == 0:
        start_time = time.time()
        cleanFrame = frame.copy()
        boxes, points_list = detector.detect(frame)

        aligned_faces = []
        embeddings = []
        detectedFaces = []

        if boxes is not None and len(boxes) > 0:
            for box, kps in zip(boxes, points_list):
                x1, y1, x2, y2, score = box
                detectedFaces.append([int(x1), int(y1), int(x2), int(y2), float(score)])
                faceKeypoints[score] = kps

            detectedFaces = np.array(detectedFaces, dtype=np.float32)

            # Tracking
            tracks = tracker.update(
                detectedFaces,
                [frame.shape[0], frame.shape[1]],  # img_info (height, width)
                [frame.shape[0], frame.shape[1]],  # img_size (height, width)
                mot20=False,
                match_thresh=trackerMatchThresh
            )

            for face in tracks:
                x1, y1, x2, y2 = map(int, face.tlbr)
                faceId = face.track_id
                kps = faceKeypoints[face.score]

                if faceId not in faceTracked:
                    faceTracked[faceId] = deque(maxlen=10)

                # ====================
                # Face Recognition
                # ====================
                emb = recognizer(cleanFrame, kps)  # Ambil 
                faceTracked[faceId].append(emb)

                if len(faceTracked[faceId]) >= 5:
                    mean_embedding = np.mean(faceTracked[faceId], axis=0)
                    sims = [compare_embeddings(mean_embedding, d["embedding"]) for d in FaceDB] if FaceDB else []
                    best_sim = max(sims) if sims else 0
                    best_idx = np.argmax(sims) if sims else -1
                    name = None
                    print(f"match : {best_sim}, name : {FaceDB[best_idx]['name']}")
                    if best_sim >= MATCH_THRESHOLD:
                        name = FaceDB[best_idx]["name"]
                    else:
                        name = "unknown"
                        # save_path = os.path.join(UNKNOWN_FOLDER, f"unknown-{unknownNumber}.jpeg")
                        # cv2.imwrite(save_path, frame)
                        # print(f"[UNKNOWN] Saved new face: {save_path}")
                        # unknownNumber += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    

        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
        # ====================
        # Tampilkan FPS
        # ====================
        cv2.putText(frame, f"{elapsed_ms:.1f} ms ({fps:.1f} FPS)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("FaceDetection", frame)

        # ====================
        # Tampilkan aligned faces
        # ====================
        if len(aligned_faces) > 0:
            grid = np.hstack([cv2.resize(face, (112, 112)) for face in aligned_faces])
            cv2.imshow("Aligned Faces", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    frameCounting += 1
    if frameCounting >= 100:
        frameCounting = 0

cap.release()
cv2.destroyAllWindows()
