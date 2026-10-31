import cv2
import numpy as np
import math
import time
import onnxruntime as ort


class YOLOv8_face:
    def __init__(self, path, conf_thres=0.45, iou_thres=0.5):
        self.conf_threshold = float(conf_thres)
        self.iou_threshold = float(iou_thres)
        self.class_names = ['face']
        self.num_classes = len(self.class_names)

        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16
        self.project = np.arange(self.reg_max, dtype=np.float32)
        self.strides = (8, 16, 32)
        self.feats_hw = [
            (math.ceil(self.input_height / stride),
             math.ceil(self.input_width / stride))
            for stride in self.strides
        ]
        self.anchors = self.make_anchors(self.feats_hw)

        # Landmark coordinate transform templates.
        self._kpt_pad = np.zeros(15, dtype=np.float32)
        self._kpt_scale = np.ones(15, dtype=np.float32)
        self._kpt_x_idx = np.arange(0, 15, 3)
        self._kpt_y_idx = np.arange(1, 15, 3)

        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active_providers = self.session.get_providers()
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"Available providers: {available_providers}")
        print(f"Session providers: {active_providers}")
        print(f"YOLO execution provider priority: {active_providers[0]}")

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Build stride-specific anchor points once, using float32."""
        anchor_points = {}
        offset = np.float32(grid_cell_offset)

        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(w, dtype=np.float32) + offset
            y = np.arange(h, dtype=np.float32) + offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)

        return anchor_points

    @staticmethod
    def softmax(x, axis=-1):
        """Numerically stable softmax."""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def resize_image(self, srcimg, keep_ratio=True):
        top, left = 0, 0
        newh, neww = self.input_height, self.input_width

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]

            if hw_scale > 1:
                newh = self.input_height
                neww = int(self.input_width / hw_scale)
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA
                )
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img, 0, 0, left, self.input_width - neww - left,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            else:
                newh = int(self.input_height * hw_scale)
                neww = self.input_width
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA
                )
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img, top, self.input_height - newh - top, 0, 0,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
        else:
            img = cv2.resize(
                srcimg, (self.input_width, self.input_height),
                interpolation=cv2.INTER_AREA
            )

        return img, newh, neww, top, left

    def detect(self, srcimg, show_timing=True):
        """Run detection and optionally print stage latency in milliseconds."""
        total_start = time.perf_counter()

        # 1. Preprocessing (kept functionally equivalent to the baseline).
        rgb_img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        input_img, newh, neww, padh, padw = self.resize_image(rgb_img)

        scale_h = srcimg.shape[0] / newh
        scale_w = srcimg.shape[1] / neww

        blob = (input_img.astype(np.float32) / 255.0)
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        blob = np.ascontiguousarray(blob, dtype=np.float32)
        preprocess_end = time.perf_counter()

        # 2. ONNX Runtime inference.
        outputs = self.session.run(None, {self.input_name: blob})
        inference_end = time.perf_counter()

        # 3. Optimized CPU post-processing.
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(
            outputs, scale_h, scale_w, padh, padw
        )
        postprocess_end = time.perf_counter()

        if show_timing:
            preprocess_ms = (preprocess_end - total_start) * 1000.0
            inference_ms = (inference_end - preprocess_end) * 1000.0
            postprocess_ms = (postprocess_end - inference_end) * 1000.0
            total_ms = (postprocess_end - total_start) * 1000.0
            print(
                f"Preprocess: {preprocess_ms:.2f} ms | "
                f"Inference: {inference_ms:.2f} ms | "
                f"Post-process: {postprocess_ms:.2f} ms | "
                f"Total: {total_ms:.2f} ms | Faces: {len(det_bboxes)}"
            )

        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        """
        Decode only candidates whose class confidence exceeds the threshold.

        The baseline decoded DFL boxes and all five landmarks for every grid
        location (8,400 locations at 640x640), then discarded low-confidence
        locations. This version filters first and decodes only retained rows.
        Assumes the batch size is 1, as in the original implementation.
        """
        all_bboxes = []
        all_scores = []
        all_class_ids = []
        all_landmarks = []

        # Reuse templates, with per-call copies so transforms cannot leak.
        kpt_pad = self._kpt_pad.copy()
        kpt_scale = self._kpt_scale.copy()
        kpt_pad[self._kpt_x_idx] = padw
        kpt_pad[self._kpt_y_idx] = padh
        kpt_scale[self._kpt_x_idx] = scale_w
        kpt_scale[self._kpt_y_idx] = scale_h
        kpt_pad = kpt_pad.reshape(1, 15)
        kpt_scale = kpt_scale.reshape(1, 15)

        bbox_pad = np.array(
            [padw, padh, padw, padh], dtype=np.float32
        ).reshape(1, 4)
        bbox_scale = np.array(
            [scale_w, scale_h, scale_w, scale_h], dtype=np.float32
        ).reshape(1, 4)

        for pred in preds:
            # Original output layout: (1, channels, height, width).
            if pred.ndim != 4 or pred.shape[0] != 1:
                raise ValueError(
                    f"Expected YOLO output shape (1,C,H,W), got {pred.shape}"
                )

            stride = int(self.input_height / pred.shape[2])
            if stride not in self.anchors:
                raise ValueError(
                    f"Unexpected YOLO feature-map height {pred.shape[2]} "
                    f"(derived stride {stride})"
                )

            # Convert to (number_of_grid_points, channels), no full decode yet.
            pred_rows = pred[0].transpose(1, 2, 0).reshape(-1, pred.shape[1])
            box_end = self.reg_max * 4
            cls_end = pred_rows.shape[1] - 15

            if cls_end <= box_end:
                raise ValueError(
                    f"Unexpected YOLO output channel count: {pred.shape[1]}"
                )

            cls_logits = pred_rows[:, box_end:cls_end]
            # Sigmoid is monotonic, so threshold logits before calculating
            # sigmoid values. This avoids computing scores for rejected rows.
            threshold = self.conf_threshold
            if threshold <= 0.0:
                candidate_indices = np.arange(pred_rows.shape[0])
            elif threshold >= 1.0:
                candidate_indices = np.empty(0, dtype=np.int64)
            else:
                logit_threshold = np.log(threshold / (1.0 - threshold))
                candidate_indices = np.flatnonzero(
                    np.max(cls_logits, axis=1) > logit_threshold
                )

            if candidate_indices.size == 0:
                continue

            selected = pred_rows[candidate_indices]
            selected_cls_logits = selected[:, box_end:cls_end]
            selected_scores = 1.0 / (1.0 + np.exp(-selected_cls_logits))
            class_ids = np.argmax(selected_scores, axis=1)
            confidences = np.max(selected_scores, axis=1)

            # DFL decode only selected candidate rows.
            box_logits = selected[:, :box_end].reshape(
                -1, 4, self.reg_max
            )
            dfl_prob = self.softmax(box_logits, axis=-1)
            distances = np.matmul(dfl_prob, self.project).reshape(-1, 4)

            selected_anchors = self.anchors[stride][candidate_indices]
            # Keep the original decoder's coordinate order exactly: decode
            # and clip in feature-map coordinates, then multiply by stride.
            boxes = self.distance2bbox(
                selected_anchors, distances,
                max_shape=(self.input_height, self.input_width)
            ) * np.float32(stride)
            boxes = (boxes - bbox_pad) * bbox_scale

            # Decode landmarks with the original full-grid indexing, then
            # retain candidate rows. This keeps landmark/anchor alignment
            # identical to the baseline while DFL decoding is candidate-only.
            kpts_all = pred_rows[:, -15:].copy().reshape(-1, 15)
            all_anchors = self.anchors[stride]
            kpts_all[:, 0::3] = (
                kpts_all[:, 0::3] * 2.0
                + (all_anchors[:, 0].reshape(-1, 1) - 0.5)
            ) * stride
            kpts_all[:, 1::3] = (
                kpts_all[:, 1::3] * 2.0
                + (all_anchors[:, 1].reshape(-1, 1) - 0.5)
            ) * stride
            kpts_all[:, 2::3] = 1.0 / (1.0 + np.exp(-kpts_all[:, 2::3]))
            kpts_all = (kpts_all - kpt_pad) * kpt_scale
            kpts = kpts_all[candidate_indices]

            # Convert xyxy to xywh for OpenCV NMS, matching the old code.
            boxes_wh = boxes.copy()
            boxes_wh[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]

            all_bboxes.append(boxes_wh)
            all_scores.append(confidences.astype(np.float32, copy=False))
            all_class_ids.append(class_ids.astype(np.int64, copy=False))
            all_landmarks.append(kpts)

        if not all_bboxes:
            empty = np.array([])
            return empty, empty, empty, empty

        bboxes_wh = np.concatenate(all_bboxes, axis=0)
        confidences = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0)

        # NMS remains OpenCV CPU NMS to keep the baseline behaviour.
        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(), confidences.tolist(),
            self.conf_threshold, self.iou_threshold
        )

        if len(indices) > 0:
            indices = np.asarray(indices).reshape(-1)
            return (
                bboxes_wh[indices],
                confidences[indices],
                class_ids[indices],
                landmarks[indices]
            )

        empty = np.array([])
        return empty, empty, empty, empty

    @staticmethod
    def distance2bbox(points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]

        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])

        return np.stack((x1, y1, x2, y2), axis=-1)