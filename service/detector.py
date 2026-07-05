import time
import threading
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import MODEL_CONFIGS, YOLO_CONFIDENCE, YOLO_INPUT_SIZE


class Detector:
    def __init__(self, model_name=None, confidence=None, core_mask=None):
        self.confidence = confidence or YOLO_CONFIDENCE
        self.nms_threshold = 0.45
        self.input_size = YOLO_INPUT_SIZE
        self.rknn = None
        self.model_path = None
        self.model_name = None
        self.classes = []
        self.last_infer_ms = 0
        self.stopped = False
        self._lock = threading.Lock()
        self._font = None
        self._core_mask = core_mask  # None 表示使用全部核心（0,1,2）
        if model_name:
            self.switch_model(model_name)

    def switch_model(self, model_name):
        if model_name == self.model_name and self.rknn is not None:
            return
        cfg = MODEL_CONFIGS.get(model_name)
        if cfg is None:
            raise ValueError(f"未知模型: {model_name}")
        self.release()
        self.model_name = model_name
        self.model_path = cfg["path"]
        self.classes = cfg["classes"]
        self.stopped = False

    def stop(self):
        self.stopped = True

    def resume(self):
        self.stopped = False

    def load_model(self):
        if self.rknn is None:
            import os
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            from rknnlite.api import RKNNLite
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                self.rknn.release()
                self.rknn = None
                raise RuntimeError(f"RKNN 模型加载失败: {self.model_path}")
            core_mask = self._core_mask if self._core_mask is not None else RKNNLite.NPU_CORE_0_1_2
            ret = self.rknn.init_runtime(core_mask=core_mask)
            if ret != 0:
                self.rknn.release()
                self.rknn = None
                raise RuntimeError("RKNN 运行时初始化失败")
        return self.rknn

    def _letterbox(self, frame, color=(0, 0, 0)):
        h, w = frame.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        dw = (self.input_size - new_w) / 2
        dh = (self.input_size - new_h) / 2

        if w != new_w or h != new_h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = int(round(dh - 0.1))
        bottom = self.input_size - new_h - top
        left = int(round(dw - 0.1))
        right = self.input_size - new_w - left

        if top > 0 or bottom > 0 or left > 0 or right > 0:
            frame = cv2.copyMakeBorder(frame, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=color)
        return frame, (r, r), (left, top)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def _dfl(self, position):
        n, c, h, w = position.shape
        mc = c // 4
        y = position.reshape(n, 4, mc, h, w)
        e_y = np.exp(y - y.max(axis=2, keepdims=True))
        y = e_y / e_y.sum(axis=2, keepdims=True)
        acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        return (y * acc).sum(2)

    def _cxcywh_to_xyxy(self, boxes_raw):
        if len(boxes_raw) == 0:
            return None
        cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    def _box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
        row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_size // grid_h, self.input_size // grid_w],
                          dtype=np.float32).reshape(1, 2, 1, 1)

        position = self._dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        class_pos = np.where(class_max_score * box_confidences >= self.confidence)
        return boxes[class_pos], classes[class_pos], (class_max_score * box_confidences)[class_pos]

    def _nms_boxes(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(ovr <= self.nms_threshold)[0] + 1]

        return np.array(keep)

    def _detect_format(self, outputs):
        if len(outputs) == 6:
            return "yolov8_branch"
        if len(outputs) == 1:
            output = outputs[0]
            if output.ndim == 3:
                output = output[0]
            rows, cols = output.shape
            num_classes = len(self.classes)
            if cols == 5 + num_classes:
                return "yolov5"
            if rows == 4 + num_classes or rows == 64 + num_classes:
                return "yolov8_single"
            if cols > rows:
                return "yolov8_single"
        return "unknown"

    def _postprocess_v8_branch(self, outputs):
        boxes, scores, classes_conf = [], [], []
        default_branch = 3
        pair_per_branch = len(outputs) // default_branch

        for i in range(default_branch):
            boxes.append(self._box_process(outputs[pair_per_branch * i]))
            classes_conf.append(outputs[pair_per_branch * i + 1])
            scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        boxes = np.concatenate([sp_flatten(v) for v in boxes])
        classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
        scores = np.concatenate([sp_flatten(v) for v in scores])

        boxes, classes, scores = self._filter_boxes(boxes, scores, classes_conf)
        return self._nms_per_class(boxes, classes, scores)

    def _postprocess_v8_single(self, output):
        num_classes = len(self.classes)
        total_features = 4 + num_classes

        if output.shape[0] == total_features and output.shape[1] != total_features:
            output = np.transpose(output).copy()

        boxes_raw = output[:, :4].astype(np.float32)
        class_scores = output[:, 4:].astype(np.float32)

        boxes_max = boxes_raw.max()
        if boxes_max > 0 and boxes_max < 1.0:
            boxes_raw *= self.input_size
        elif boxes_max > self.input_size:
            boxes_raw = boxes_raw / boxes_max * self.input_size

        class_max_score = np.max(class_scores, axis=-1)
        classes = np.argmax(class_scores, axis=-1)

        mask = class_max_score >= self.confidence
        boxes_raw = boxes_raw[mask]
        classes = classes[mask]
        scores = class_max_score[mask]

        boxes = self._cxcywh_to_xyxy(boxes_raw)
        if boxes is None:
            return None, None, None

        return self._nms_per_class(boxes, classes, scores)

    def _postprocess_v5(self, outputs):
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]

        num_classes = len(self.classes)
        expected_cols = 5 + num_classes

        if output.shape[1] < expected_cols:
            return None, None, None

        if output.dtype == np.uint8:
            output = output.astype(np.float32) / 255.0

        boxes_raw = output[:, :4].astype(np.float32)
        obj_conf = self._sigmoid(output[:, 4].astype(np.float32))
        class_scores = self._sigmoid(output[:, 5:5 + num_classes].astype(np.float32))

        class_max_score = np.max(class_scores, axis=-1)
        classes = np.argmax(class_scores, axis=-1)
        final_scores = obj_conf * class_max_score

        mask = final_scores >= self.confidence
        boxes_raw = boxes_raw[mask]
        classes = classes[mask]
        scores = final_scores[mask]

        boxes = self._cxcywh_to_xyxy(boxes_raw)
        if boxes is None:
            return None, None, None

        return self._nms_per_class(boxes, classes, scores)

    def _nms_per_class(self, boxes, classes, scores):
        nboxes, nclasses, nscores = [], [], []
        for c in np.unique(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            s = scores[inds]
            keep = self._nms_boxes(b, s)
            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(np.full(len(keep), c))
                nscores.append(s[keep])

        if not nboxes:
            return None, None, None

        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def _postprocess(self, outputs):
        if outputs is None or len(outputs) == 0:
            return None, None, None

        fmt = self._detect_format(outputs)

        if fmt == "yolov8_branch":
            return self._postprocess_v8_branch(outputs)
        elif fmt == "yolov5":
            return self._postprocess_v5(outputs)
        else:
            output = outputs[0]
            if output.ndim == 3:
                output = output[0]
            return self._postprocess_v8_single(output)

    def predict(self, frame):
        if self.stopped:
            return []

        with self._lock:
            rknn = self.load_model()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img, ratio, padding = self._letterbox(img_rgb)
            img = np.expand_dims(img, 0)

            t0 = time.time()
            outputs = rknn.inference(inputs=[img], data_format=['nhwc'])
            self.last_infer_ms = round((time.time() - t0) * 1000, 1)

            if self.stopped:
                return []

            boxes, classes, scores = self._postprocess(outputs)

        detections = []
        if boxes is not None:
            h, w = frame.shape[:2]
            px, py = padding[0], padding[1]
            rx, ry = ratio[0], ratio[1]

            for box, score, cl in zip(boxes, scores, classes):
                if self.stopped:
                    return []

                x1 = max(0, int((box[0] - px) / rx))
                y1 = max(0, int((box[1] - py) / ry))
                x2 = min(w, int((box[2] - px) / rx))
                y2 = min(h, int((box[3] - py) / ry))

                cls_id = int(cl)
                label = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                detections.append({
                    "label": label,
                    "confidence": round(float(score), 2),
                    "bbox": [x1, y1, x2, y2],
                })

        return detections

    def _load_font(self):
        if self._font is not None:
            return self._font
        for fp in (
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "simhei.ttf",
            "msyh.ttc",
        ):
            try:
                self._font = ImageFont.truetype(fp, 20)
                return self._font
            except (IOError, OSError):
                continue
        self._font = ImageFont.load_default()
        return self._font

    def predict_and_draw(self, frame):
        try:
            detections = self.predict(frame)
        except Exception as e:
            return frame, None
        if not detections:
            return frame, None

        font = self._load_font()
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]

            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            draw.rectangle([x1, y1 - th - 8, x1 + tw + 4, y1], fill=(0, 255, 0))
            draw.text((x1 + 2, y1 - th - 4), text, fill=(0, 0, 0), font=font)

        annotated = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        info = [{"label": d["label"], "confidence": d["confidence"]} for d in detections]
        return annotated, info

    def release(self):
        self.stopped = True
        with self._lock:
            if self.rknn is not None:
                self.rknn.release()
                self.rknn = None
