import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis

# ======================================================
# 🔧 KONFIGURASI
# ======================================================
QUERY_IMAGE = "database/natalia-1.jpeg"
DATABASE_DIR = "database"  # Folder tempat semua foto disimpan
THRESHOLD = 0.4  # Semakin tinggi, semakin ketat (0.6–0.7 umum dipakai)

# ======================================================
# 🚀 INISIALISASI MODEL INSIGHTFACE
# ======================================================
# MODEL LIST:
# "buffalo_l"  → Large, akurat
# "buffalo_m"  → Medium, seimbang speed/akurasi
# "antelopev2" → Cepat, cukup akurat
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)  # ctx_id=-1 = CPU, 0 = GPU

# ======================================================
# ⚙️ FUNGSI UTILITAS
# ======================================================
def get_face_embedding_and_crop(image_path):
    """Ambil embedding dan hasil crop wajah"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError(f"No face detected in {image_path}")
    
    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    cropped = img[y1:y2, x1:x2]
    return face.embedding, cropped

def compare_faces(emb1, emb2):
    """Hitung cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity

# ======================================================
# 🧠 PROSES PEMBANDINGAN
# ======================================================
try:
    emb_query, crop_query = get_face_embedding_and_crop(QUERY_IMAGE)
    print(f"Query face loaded: {QUERY_IMAGE}\n")

    for filename in os.listdir(DATABASE_DIR):
        img_path = os.path.join(DATABASE_DIR, filename)

        # Skip file query sendiri
        if img_path == QUERY_IMAGE:
            continue
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            emb_db, crop_db = get_face_embedding_and_crop(img_path)
            sim = compare_faces(emb_query, emb_db)
            is_same = sim > THRESHOLD

            # Gabung dua crop untuk ditampilkan berdampingan
            h = max(crop_query.shape[0], crop_db.shape[0])
            w = crop_query.shape[1] + crop_db.shape[1]
            combined = np.zeros((h, w, 3), dtype=np.uint8)
            combined[:crop_query.shape[0], :crop_query.shape[1]] = crop_query
            combined[:crop_db.shape[0], crop_query.shape[1]:] = crop_db

            label = f"{filename} | Sim: {sim:.4f} | {'MATCH' if is_same else 'NO MATCH'}"
            cv2.putText(combined, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (0, 255, 0) if is_same else (0, 0, 255), 2)
            
            cv2.imshow("Face Comparison", combined)
            print(f"{filename:<25}  Similarity: {sim:.4f}  -->  {'MATCH' if is_same else 'NO MATCH'}")

            key = cv2.waitKey(0)
            if key == 27:  # ESC to exit
                break

        except Exception as e:
            print(f"Error on {filename}: {e}")

    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error: {str(e)}")
