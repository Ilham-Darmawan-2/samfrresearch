import os
import cv2
import numpy as np
import onnxruntime
import time  # <<=== Tambah ini

from utils.frutility import distance2bbox, distance2kps, draw_corners, draw_keypoints
from typing import Tuple

__all__ = ['SCRFD']

def create_inference_session(model_path: str):
    available_providers = onnxruntime.get_available_providers()
    print("Available providers:", available_providers)

    # Default fallback
    providers = ["CPUExecutionProvider"]

    # NVIDIA GPU (CUDA / TensorRT)
    if "CUDAExecutionProvider" in available_providers:
        providers = [
            ('TensorrtExecutionProvider', {'trt_engine_cache_enable': True}),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        print("🟢 Using NVIDIA GPU (TensorRT + CUDA)")

    # Intel GPU (OpenVINO)
    elif "OpenVINOExecutionProvider" in available_providers:
        providers = [
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider'
        ]
        print("🔵 Using Intel GPU (OpenVINO)")

    # (Opsional) DirectML / ROCm
    elif "DmlExecutionProvider" in available_providers:
        providers = [
            'DmlExecutionProvider',
            'CPUExecutionProvider'
        ]
        print("🟣 Using DirectML (Windows GPU)")

    elif "ROCMExecutionProvider" in available_providers:
        providers = [
            'ROCMExecutionProvider',
            'CPUExecutionProvider'
        ]
        print("🔴 Using AMD GPU (ROCm)")

    else:
        print("⚪ No GPU provider found — using CPU only")

    session = onnxruntime.InferenceSession(model_path, providers=providers)
    print("✅ Active providers:", session.get_providers())
    return session

class SCRFD:
    """
    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4
    ) -> None:

        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # SCRFD model params --------------
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}
        # ---------------------------------

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        try:
            # self.session = onnxruntime.InferenceSession(
            #     model_path,
            #     providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            # )
            self.session = create_inference_session(model_path=model_path)
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def forward(self, image, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(image.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, image, max_num=1, metric="max"):
        width, height = self.input_size

        im_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = height / width
        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))

        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image

        scores_list, bboxes_list, kpss_list = self.forward(det_image, self.conf_thres)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale

        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det, iou_thres=self.iou_thres)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack(
                [
                    (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - image_center[0],
                ]
            )
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = (area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets, iou_thres):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= iou_thres)[0]
            order = order[indices + 1]

        return keep


    def align_face(self, image, keypoints, output_size=(112, 112)):
        """Align wajah berdasarkan 5 keypoints"""
        # urutan keypoints: [left_eye, right_eye, nose, mouth_left, mouth_right]
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        dst = keypoints.astype(np.float32)

        # Estimasi transformasi similarity
        tform, _ = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
        if tform is None:
            return None

        aligned = cv2.warpAffine(image, tform, output_size, borderValue=0.0)
        return aligned

if __name__ == "__main__":
    detector = SCRFD(model_path="models/buffalo_m/det_25g.onnx")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        boxes, points_list = detector.detect(frame)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0

        aligned_faces = []

        if boxes is not None and len(boxes) > 0:
            for box, kps in zip(boxes, points_list):
                x1, y1, x2, y2, score = box.astype(np.int32)
                draw_corners(frame, (x1, y1, x2, y2))
                draw_keypoints(frame, kps)

                aligned = detector.align_face(frame, kps)
                if aligned is not None:
                    aligned_faces.append(aligned)

        cv2.putText(frame, f"{elapsed_ms:.1f} ms ({fps:.1f} FPS)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("FaceDetection", frame)

        # tampilkan hasil alignment di window lain
        if len(aligned_faces) > 0:
            grid = np.hstack([cv2.resize(face, (112, 112)) for face in aligned_faces])
            cv2.imshow("Aligned Faces", grid)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

