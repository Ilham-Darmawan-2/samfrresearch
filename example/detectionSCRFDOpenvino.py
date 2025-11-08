import os
import cv2
import numpy as np
import onnxruntime
import time
from typing import Tuple
from utils.frutility import distance2bbox, distance2kps, draw_corners, draw_keypoints

__all__ = ['SCRFD']


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

        # SCRFD params
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.mean = 127.5
        self.std = 128.0

        self.center_cache = {}
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """Initialize the model and handle dynamic input shapes."""
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self.output_names = [x.name for x in self.session.get_outputs()]
            self.input_names = [x.name for x in self.session.get_inputs()]
            input_shape = self.session.get_inputs()[0].shape

            # Handle dynamic input shapes like [None, 3, None, None]
            if None in input_shape or 'None' in str(input_shape):
                print("[INFO] Dynamic input shape detected — using fixed shape (1, 3, H, W)")
                self.fixed_shape = [1, 3, self.input_size[1], self.input_size[0]]
            else:
                self.fixed_shape = input_shape
        except Exception as e:
            print(f"[ERROR] Failed to load the model: {e}")
            raise

    def forward(self, image, threshold):
        if image is None or image.size == 0:
            raise ValueError("[ERROR] Image is empty!")

        scores_list, bboxes_list, kpss_list = [], [], []

        # Pakai input_size yang sudah kita tentukan
        input_width, input_height = self.input_size
        input_size = (input_width, input_height)

        # resize image supaya sesuai model
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, input_size),
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )

        outputs = self.session.run(self.output_names, {self.input_names[0]: blob})

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride
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

    def detect(self, image, max_num=0, metric="max"):
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
            kpss = kpss[order, :, :][keep, :, :]
        else:
            kpss = None

        if 0 < max_num < det.shape[0]:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            image_center = image.shape[0] // 2, image.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - image_center[1],
                (det[:, 1] + det[:, 3]) / 2 - image_center[0],
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == "max" else (area - offset_dist_squared * 2.0)
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

    def nms(self, dets, iou_thres):
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
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

    # ======================
    # Face Alignment Utility
    # ======================
    def align_face(self, image, keypoints, output_size=(112, 112)):
        # standard 5-point reference (ArcFace style)
        ref_pts = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        dst = ref_pts
        src = np.array(keypoints, dtype=np.float32)
        tform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
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
        dets, kpss = detector.detect(frame)
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0

        aligned_faces = []

        if dets is not None and len(dets) > 0:
            for det, kps in zip(dets, kpss):
                x1, y1, x2, y2, score = det.astype(np.int32)
                draw_corners(frame, (x1, y1, x2, y2))
                draw_keypoints(frame, kps)
                aligned = detector.align_face(frame, kps)
                aligned_faces.append(aligned)

        cv2.putText(frame, f"{elapsed_ms:.1f} ms ({fps:.1f} FPS)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("FaceDetection", frame)

        # tampilkan hasil alignment di window lain
        if len(aligned_faces) > 0:
            stack = np.hstack([cv2.resize(f, (112, 112)) for f in aligned_faces])
            cv2.imshow("Aligned Faces", stack)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
