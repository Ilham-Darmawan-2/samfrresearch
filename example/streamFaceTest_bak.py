import cv2
import os
import time
import numpy as np
from openvino import Core
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess
from utils.byte_tracker import BYTETracker
from insightface.app import FaceAnalysis

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "models/yolox_face/model.xml"
LABELS_PATH = "models/yolox_face/labels.txt"
VIDEO_PATH = 0
DATABASE_PATH = "database"
UNKNOWN_FOLDER = "unknown"

DEVICE = "CPU"
FRAME_RESIZE = (1280, 720)

SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.45
TRACK_THRESH = 0.5
TRACK_BUFFER = 30
MATCH_THRESHOLD = 0.5
SKIPFRAME = 5
RECOOLDOWN = 3  # detik


# ======================================================
# 🚀 INISIALISASI MODEL INSIGHTFACE
# ======================================================
# MODEL LIST:
# "buffalo_l"  → Large, akurat
# "buffalo_m"  → Medium, seimbang speed/akurasi
# "antelopev2" → Cepat, cukup akurat
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)  # ctx_id=-1 = CPU, 0 = GPU


# =====================================================
# UTILS
# =====================================================
def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


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


# =====================================================
# FACE DATABASE (pakai YOLOX + InsightFace embedding)
# =====================================================
def build_face_database(folder_path):

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

        # Dapat embedding dari InsightFace
        faces = app.get(img)
        if len(faces) == 0:
            print(f"[ERROR] {img_name}: no embedding extracted")
            continue

        emb = faces[0].embedding
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


# =====================================================
# MAIN LOOP
# =====================================================
def main():
    core = Core()
    model = core.read_model(MODEL_PATH)
    model.reshape({model.inputs[0].any_name: [1, 3, 416, 416]})
    compiled_model = core.compile_model(model=model, device_name=DEVICE)
    input_blob = compiled_model.inputs[0].any_name
    output_blob = compiled_model.outputs[0].any_name
    _, _, input_h, input_w = compiled_model.input(0).shape

    tracker = BYTETracker(track_thresh=TRACK_THRESH, track_buffer=TRACK_BUFFER)

    FaceDB = build_face_database(DATABASE_PATH)
    unknownNumber = get_last_unknown_number(UNKNOWN_FOLDER) + 1
    recognized_cache = {}

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Stream ended or failed.")
            continue

        frame_num += 1
        if frame_num % SKIPFRAME != 0:
            continue

        start_time = time.time()
        processed_img, ratio = preprocess(frame, (input_h, input_w))
        processed_img = np.expand_dims(processed_img, axis=0)
        results = compiled_model([processed_img])[output_blob]
        predictions = demo_postprocess(results, (input_h, input_w))[0]
        bboxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        bboxes_xyxy = np.zeros_like(bboxes)
        bboxes_xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        bboxes_xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        bboxes_xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes_xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        bboxes_xyxy /= ratio

        detections = multiclass_nms(bboxes_xyxy, scores, nms_thr=NMS_THRESHOLD, score_thr=SCORE_THRESHOLD)

        if detections is not None and len(detections) > 0:
            dets = []
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                dets.append([x1, y1, x2, y2, conf])
            dets = np.array(dets)

            tracks = tracker.update(
                dets, 
                [frame.shape[0], frame.shape[1]],
                [frame.shape[0], frame.shape[1]], 
                mot20=False,
                match_thresh=MATCH_THRESHOLD
            )

            for t in tracks:
                x1, y1, x2, y2 = map(int, t.tlbr)
                tid = t.track_id
                now = time.time()
                name_display = f"id:{tid}"

                if tid in recognized_cache and now - recognized_cache[tid]["last_time"] < RECOOLDOWN:
                    name_display = f"{recognized_cache[tid]['name']} (id:{tid})"
                else:
                    faces = app.get(frame)

                    if len(faces) == 0:
                        continue

                    emb = faces[0].embedding
                    sims = [compare_embeddings(emb, d["embedding"]) for d in FaceDB] if FaceDB else []
                    best_sim = max(sims) if sims else 0
                    best_idx = np.argmax(sims) if sims else -1

                    print(f"match : {best_sim}, name : {FaceDB[best_idx]['name']}")
                    if best_sim >= MATCH_THRESHOLD:
                        name = FaceDB[best_idx]["name"]
                    else:
                        name = "unknown"
                        save_path = os.path.join(UNKNOWN_FOLDER, f"unknown-{unknownNumber}.jpeg")
                        cv2.imwrite(save_path, frame)
                        print(f"[UNKNOWN] Saved new face: {save_path}")
                        unknownNumber += 1

                    recognized_cache[tid] = {"name": name, "last_time": now}
                    name_display = f"{name} (id:{tid})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name_display, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS / inference time
        inf_ms = (time.time() - start_time) * 1000
        disp = cv2.resize(frame, FRAME_RESIZE)
        cv2.putText(disp, f"Inference: {inf_ms:.1f} ms", (700, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Recognition (YOLOX + InsightFace)", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
