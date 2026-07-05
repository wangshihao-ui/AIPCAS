"""
每个 NPU 核心独占一个进程

通信协议核心:
  frame_queue: 主进程 → 子进程，发送 (frame_bgr, gen) 或 None(退出)
  result_queue: 子进程 → 主进程，发送 (annotated_frame, info, infer_ms, gen)
  stop_event: 主进程通知子进程退出
"""
import os
import sys

# spawn 子进程不会继承父进程的 sys.path，需手动添加项目根目录
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np


def inference_worker(model_name, core_mask, confidence, nms_threshold,
                     frame_queue, result_queue, stop_event):
    """在子进程中加载 RKNN 模型并执行推理"""
    from service.detector import Detector

    try:
        detector = Detector(
            model_name=model_name,
            confidence=confidence,
            core_mask=core_mask,
        )
        detector.nms_threshold = nms_threshold
    except Exception as e:
        # 模型加载/初始化失败，通知主进程并退出
        result_queue.put((None, None, 0, -1))  # gen=-1 表示初始化失败
        print(f"[Worker] 模型加载失败: {e}")
        return

    while not stop_event.is_set():
        try:
            item = frame_queue.get(timeout=0.5)
        except Exception:
            continue

        if item is None:
            break

        frame, gen = item
        try:
            annotated, info = detector.predict_and_draw(frame)
            ms = detector.last_infer_ms
            result_queue.put((annotated, info, ms, gen))
        except Exception:
            result_queue.put((frame, None, 0, gen))

    detector.release()
