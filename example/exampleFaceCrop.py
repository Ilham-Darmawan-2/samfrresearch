import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Inisialisasi model InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1)

def get_face_and_crop(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No faces detected")
    
    # Ambil wajah pertama
    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)

    # Crop wajah
    cropped_face = img[y1:y2, x1:x2]

    # Tampilkan hasil crop dan kotaknya
    img_boxed = img.copy()
    cv2.rectangle(img_boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Original with Box", img_boxed)
    cv2.imshow("Cropped Face", cropped_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_face

# Jalankan
image_path = "database/ilham-6.jpeg"
get_face_and_crop(image_path)
