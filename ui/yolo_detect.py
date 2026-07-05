import os
import time
import threading
import queue
import multiprocessing as mp
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QComboBox, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QMessageBox, QFrame, QGraphicsDropShadowEffect, QScrollArea,
    QSlider, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
import qtawesome as qta
from service.detector import Detector
from service.inference_worker import inference_worker
from config import (
    MODEL_CONFIGS, CAMERA_DEVICE_1, CAMERA_DEVICE_2,
    CAMERA_WIDTH, CAMERA_HEIGHT, BASE_DIR, ENABLE_EFFECTS,
)
from utils.record_manager import (
    add_record as rm_add_record,
    load_records as rm_load_records,
    clear_records as rm_clear_records,
)

SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# 常量

FONT_SIZE_RECORD_TIME    = 14
FONT_SIZE_RECORD_TYPE    = 14
FONT_SIZE_PLACEHOLDER    = 18
FONT_SIZE_BTN            = 14
ICON_SIZE_BTN            = 20
ICON_SIZE_CLEAR_BTN      = 20
ICON_SIZE_JUMP           = 20
FONT_SIZE_CAMERA_NAME     = 18
FONT_SIZE_CAMERA_STATUS   = 15

BG_COMBOBOX    = "rgba(255, 255, 255, 0.12)"
BG_COMBOBOX_LIST = "rgba(40, 40, 60, 0.9)"
BG_SELECTION   = "rgba(0, 122, 255, 0.35)"
BG_CARD_LIGHT  = "rgba(255, 255, 255, 0.12)"
BRD_CARD_LIGHT = "rgba(255, 255, 255, 0.18)"
BRD_CARD_DARK  = "rgba(255, 255, 255, 0.15)"
BRD_INPUT      = "rgba(255, 255, 255, 0.25)"


def _shadow(blur=30, x=0, y=8, color=None):
    if not ENABLE_EFFECTS:
        return None
    if color is None:
        color = QColor(0, 0, 0, 80)
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setXOffset(x)
    s.setYOffset(y)
    s.setColor(color)
    return s


def _bg(c1, c2):
    if ENABLE_EFFECTS:
        return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {c1}, stop:1 {c2})"
    return c1


def _card_style(radius):
    return f"""
    QFrame {{
        background: {BG_CARD_LIGHT};
        border: 1px solid {BRD_CARD_LIGHT};
        border-radius: {radius}px;
    }}
"""


# ============================================================
# 记录项
# ============================================================
class RecordItem(QFrame):
    clicked = pyqtSignal()

    def __init__(self, timestamp, pest_type, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pest_type = pest_type
        self.timestamp = timestamp
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 14px;
            }
            QFrame:hover {
                background: #f5f5f5;
                border: 1px solid #cccccc;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(70, 50)
        self.thumb_label.setStyleSheet("""
            QLabel { border-radius: 8px; background: #f0f0f0; }
        """)
        self.thumb_label.setScaledContents(True)
        layout.addWidget(self.thumb_label)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)

        self.time_label = QLabel(timestamp)
        self.time_label.setFont(QFont("SF Pro Display", FONT_SIZE_RECORD_TIME))
        self.time_label.setStyleSheet("color: #666666;")
        info_layout.addWidget(self.time_label)

        self.type_label = QLabel(pest_type)
        self.type_label.setFont(QFont("SF Pro Display", FONT_SIZE_RECORD_TYPE, QFont.Bold))
        self.type_label.setStyleSheet("color: #333333;")
        info_layout.addWidget(self.type_label)

        layout.addLayout(info_layout, 1)

        self.jump_icon = QLabel()
        self.jump_icon.setPixmap(
            qta.icon('fa5s.external-link-alt', color='#999999')
            .pixmap(ICON_SIZE_JUMP, ICON_SIZE_JUMP)
        )
        layout.addWidget(self.jump_icon)

    def set_thumbnail(self, pixmap):
        if pixmap and not pixmap.isNull():
            self.thumb_label.setPixmap(
                pixmap.scaled(70, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def mousePressEvent(self, event):
        self.clicked.emit()



# 摄像头组件

class _CameraFeed(QFrame):
    """封装一路摄像头的显示区域、模型选择、状态指示、参数滑块和启停按钮"""

    _infer_done = pyqtSignal(object)

    def __init__(self, cam_id, label, device_id, core_mask, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.label = label
        self.device_id = device_id
        self._active = False
        self._capture = None
        self._is_video_file = False
        self._video_fps = 30
        self._frame_count = 0
        self._skip_frames = 2
        self._infer_gen = 0
        self._last_annotated = None

        # 参数
        self._model = list(MODEL_CONFIGS.keys())[0] if MODEL_CONFIGS else ""
        self._param_confidence = 0.60
        self._param_save_interval = 3
        self._param_nms = 0.45
        self._core_mask = core_mask   # NPU 核心分配

        # 后台取流
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_thread = None

        # 子进程推理（进程隔离，避免 stack smashing）
        self._mp_ctx = mp.get_context('spawn')
        self._worker = None
        self._frame_queue = None
        self._result_queue = None
        self._stop_event = None

        # 调参控件引用（用于启动/停止时禁用/启用）
        self._param_widgets = []

        # 显示定时器（纯显示，不阻塞 UI）
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._display_tick)

        # 子进程结果轮询定时器
        self._result_poll_timer = QTimer(self)
        self._result_poll_timer.timeout.connect(self._poll_result)

        # 检测器（独享 NPU 核）
        self._detector = Detector(
            model_name=self._model,
            confidence=self._param_confidence,
            core_mask=core_mask,
        )
        self._detector.nms_threshold = self._param_nms
        self._last_save_time = {}

        self._infer_done.connect(self._on_infer_done)

        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet(_card_style(radius=18))
        self.setGraphicsEffect(_shadow(30, 0, 10))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # ---- 标题栏 ----
        header = QHBoxLayout()
        header.setSpacing(8)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(10, 10)
        self._status_dot.setStyleSheet(
            "QLabel { background: #FF3B30; border-radius: 5px; border: none; }"
        )
        header.addWidget(self._status_dot)

        self._name_label = QLabel(self.label)
        self._name_label.setFont(QFont("SF Pro Display", FONT_SIZE_CAMERA_NAME, QFont.Bold))
        self._name_label.setStyleSheet("color: rgba(255,255,255,0.9); border:none;")
        header.addWidget(self._name_label)

        header.addStretch()

        # 模型下拉框
        self._model_combo = QComboBox()
        self._model_combo.setFont(QFont("SF Pro Display", FONT_SIZE_CAMERA_STATUS))
        self._model_combo.addItems(MODEL_CONFIGS.keys())
        self._model_combo.setMinimumWidth(180)
        self._model_combo.setMinimumHeight(32)
        self._model_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 4px 8px;
                border: 1px solid {BRD_INPUT};
                border-radius: 6px;
                background: {BG_COMBOBOX};
                color: rgba(255,255,255,0.9);
            }}
            QComboBox:hover {{ background: rgba(255,255,255,0.22); }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background: {BG_COMBOBOX_LIST};
                border: 1px solid {BRD_CARD_DARK};
                border-radius: 8px;
                color: white;
                selection-background-color: {BG_SELECTION};
                padding: 4px;
            }}
        """)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        header.addWidget(self._model_combo)

        # 启停按钮
        self._btn_toggle = QPushButton("  启动")
        self._btn_toggle.setIcon(qta.icon('fa5s.play', color='white'))
        self._btn_toggle.setIconSize(QSize(20, 20))
        self._btn_toggle.setCheckable(True)
        self._btn_toggle.setCursor(Qt.PointingHandCursor)
        self._btn_toggle.setMinimumHeight(38)
        self._btn_toggle.setMinimumWidth(110)
        self._btn_toggle.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 18px;
                border: none;
                border-radius: 10px;
                font-size: 17px;
                font-weight: bold;
                color: white;
                font-family: "SF Pro Display";
                background: {_bg('#34C759', '#30D158')};
            }}
            QPushButton:checked {{
                background: {_bg('#FF3B30', '#FF453A')};
            }}
            QPushButton:hover {{ opacity: 0.85; }}
        """)
        self._btn_toggle.toggled.connect(self._on_toggle)
        header.addWidget(self._btn_toggle)

        layout.addLayout(header)

        # ---- 视频显示区 ----
        self._display = QLabel()
        self._display.setAlignment(Qt.AlignCenter)
        self._display.setScaledContents(False)
        self._display.setMinimumSize(320, 240)
        self._display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._display.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 0, 0, 0.25);
                border: 2px dashed rgba(255, 255, 255, 0.3);
                border-radius: 14px;
                color: rgba(255, 255, 255, 0.7);
                font-size: {FONT_SIZE_PLACEHOLDER}px;
                font-family: "SF Pro Display";
            }}
        """)
        self._display.setText(f"摄像头 {self.cam_id} — 等待启动")
        layout.addWidget(self._display, 2)

        # ---- 参数栏 ----
        param_panel = QFrame()
        param_panel.setStyleSheet("QFrame { background: transparent; border: none; }")
        param_layout = QVBoxLayout(param_panel)
        param_layout.setContentsMargins(6, 4, 6, 4)
        param_layout.setSpacing(8)

        lbl_style = "color: rgba(255,255,255,0.85); border:none; font-family: 'SF Pro Display';"

        # 置信度
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        c_lbl = QLabel("置信度")
        c_lbl.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        c_lbl.setStyleSheet(lbl_style)
        c_lbl.setFixedWidth(110)
        c_lbl.setAlignment(Qt.AlignCenter)
        row2.addWidget(c_lbl)
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 95)
        self._conf_slider.setValue(60)
        self._conf_slider.setFixedWidth(140)
        self._conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: rgba(255, 255, 255, 0.15); border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #007AFF; border: 2px solid #007AFF;
                width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007AFF, stop:1 #5856D6);
                border-radius: 2px;
            }
        """)
        row2.addWidget(self._conf_slider)
        self._conf_value = QLabel("0.60")
        self._conf_value.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        self._conf_value.setStyleSheet("color: #5AC8FA; border:none; min-width: 80px; font-family: 'SF Pro Display';")
        self._conf_value.setAlignment(Qt.AlignCenter)
        row2.addWidget(self._conf_value)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        row2.addStretch()
        param_layout.addLayout(row2)

        # 保存间隔
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        i_lbl = QLabel("保存间隔")
        i_lbl.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        i_lbl.setStyleSheet(lbl_style)
        i_lbl.setFixedWidth(110)
        i_lbl.setAlignment(Qt.AlignCenter)
        row3.addWidget(i_lbl)
        self._interval_slider = QSlider(Qt.Horizontal)
        self._interval_slider.setRange(1, 30)
        self._interval_slider.setValue(3)
        self._interval_slider.setFixedWidth(140)
        self._interval_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: rgba(255, 255, 255, 0.15); border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #34C759; border: 2px solid #34C759;
                width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34C759, stop:1 #30D158);
                border-radius: 2px;
            }
        """)
        row3.addWidget(self._interval_slider)
        self._interval_value = QLabel("3s")
        self._interval_value.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        self._interval_value.setStyleSheet("color: #5AC8FA; border:none; min-width: 80px; font-family: 'SF Pro Display';")
        self._interval_value.setAlignment(Qt.AlignCenter)
        row3.addWidget(self._interval_value)
        self._interval_slider.valueChanged.connect(self._on_interval_changed)
        row3.addStretch()
        param_layout.addLayout(row3)

        # NMS
        row4 = QHBoxLayout()
        row4.setSpacing(8)
        n_lbl = QLabel("NMS")
        n_lbl.setFont(QFont("SF Pro Display", 14, QFont.Bold))
        n_lbl.setStyleSheet(lbl_style)
        n_lbl.setFixedWidth(110)
        n_lbl.setAlignment(Qt.AlignCenter)
        row4.addWidget(n_lbl)
        self._nms_slider = QSlider(Qt.Horizontal)
        self._nms_slider.setRange(10, 95)
        self._nms_slider.setValue(45)
        self._nms_slider.setFixedWidth(140)
        self._nms_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: rgba(255, 255, 255, 0.15); border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #FF9500; border: 2px solid #FF9500;
                width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF9500, stop:1 #FF6B00);
                border-radius: 2px;
            }
        """)
        row4.addWidget(self._nms_slider)
        self._nms_value = QLabel("0.45")
        self._nms_value.setFont(QFont("SF Pro Display", 16, QFont.Bold))
        self._nms_value.setStyleSheet("color: #5AC8FA; border:none; min-width: 80px; font-family: 'SF Pro Display';")
        self._nms_value.setAlignment(Qt.AlignCenter)
        row4.addWidget(self._nms_value)
        self._nms_slider.valueChanged.connect(self._on_nms_changed)
        row4.addStretch()
        param_layout.addLayout(row4)

        layout.addWidget(param_panel, 1)

        # 收集所有调参控件
        self._param_widgets = [
            self._model_combo,
            self._conf_slider, self._conf_value,
            self._interval_slider, self._interval_value,
            self._nms_slider, self._nms_value,
        ]

    def _set_params_enabled(self, enabled):
        """启用/禁用所有调参控件"""
        for w in self._param_widgets:
            w.setEnabled(enabled)

    # ---- 参数回调 ----
    def _on_model_changed(self, name):
        self._model = name

    def _on_conf_changed(self, val):
        self._param_confidence = val / 100.0
        self._conf_value.setText(f"{self._param_confidence:.2f}")
        self._detector.confidence = self._param_confidence

    def _on_interval_changed(self, val):
        self._param_save_interval = val
        self._interval_value.setText(f"{val}s")

    def _on_nms_changed(self, val):
        self._param_nms = val / 100.0
        self._nms_value.setText(f"{self._param_nms:.2f}")
        self._detector.nms_threshold = self._param_nms

    # ---- 启停 ----
    def _on_toggle(self, checked):
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self):
        if not self._model:
            self._display.setText(f"摄像头 {self.cam_id} — 未选择模型")
            self._btn_toggle.blockSignals(True)
            self._btn_toggle.setChecked(False)
            self._btn_toggle.blockSignals(False)
            return
        self._detector.stopped = False
        try:
            self._detector.switch_model(self._model)
            self._detector.confidence = self._param_confidence
            self._detector.nms_threshold = self._param_nms
        except Exception as e:
            self._display.setText(f"模型加载失败: {e}")
            self._btn_toggle.blockSignals(True)
            self._btn_toggle.setChecked(False)
            self._btn_toggle.blockSignals(False)
            return

        self._capture = cv2.VideoCapture(self.device_id)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        if not self._capture.isOpened():
            self._capture = None
            self._display.setText(f"摄像头 {self.cam_id} 无法打开 (设备 {self.device_id})")
            self._btn_toggle.blockSignals(True)
            self._btn_toggle.setChecked(False)
            self._btn_toggle.blockSignals(False)
            return

        self._is_video_file = False
        self._active = True
        self._latest_frame = None
        self._last_annotated = None
        self._last_save_time.clear()

        # 释放主进程中的模型，避免与子进程冲突
        self._detector.release()

        # 启动子进程推理
        self._start_worker()

        # 启动后台取流线程
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # 主线程定时器
        self._display_timer.start(50)
        self._result_poll_timer.start(30)

        self._set_params_enabled(False)

        self._btn_toggle.setText("  停止")
        self._btn_toggle.setIcon(qta.icon('fa5s.stop', color='white'))
        self._status_dot.setStyleSheet("QLabel { background: #34C759; border-radius: 5px; border: none; }")
        self._display.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 0, 0, 0.35);
                border: 2px solid rgba(52, 199, 89, 0.5);
                border-radius: 14px;
                color: rgba(255, 255, 255, 0.7);
                font-size: {FONT_SIZE_PLACEHOLDER}px;
                font-family: "SF Pro Display";
            }}
        """)
        self._display.setText("")

    def _stop(self):
        self._active = False
        self._infer_gen += 1
        if self._display_timer.isActive():
            self._display_timer.stop()
        if self._result_poll_timer.isActive():
            self._result_poll_timer.stop()
        if self._capture and self._capture.isOpened():
            self._capture.release()
        self._capture = None
        self._is_video_file = False
        self._latest_frame = None
        self._last_annotated = None

        # 停止子进程推理
        self._stop_worker()

        self._btn_toggle.blockSignals(True)
        self._btn_toggle.setChecked(False)
        self._btn_toggle.blockSignals(False)
        self._btn_toggle.setText("  启动")
        self._btn_toggle.setIcon(qta.icon('fa5s.play', color='white'))

        self._set_params_enabled(True)
        self._status_dot.setStyleSheet("QLabel { background: #FF3B30; border-radius: 5px; border: none; }")
        self._display.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 0, 0, 0.25);
                border: 2px dashed rgba(255, 255, 255, 0.3);
                border-radius: 14px;
                color: rgba(255, 255, 255, 0.7);
                font-size: {FONT_SIZE_PLACEHOLDER}px;
                font-family: "SF Pro Display";
            }}
        """)
        self._display.setPixmap(QPixmap())
        self._display.setText(f"摄像头 {self.cam_id} — 等待启动")

    # ---- 帧更新 ----
    def _capture_loop(self):
        """后台线程：持续从摄像头/视频取帧，推入子进程推理队列"""
        local_count = 0
        # 视频文件不跳帧每帧都送推理，摄像头按 skip_frames 跳帧
        skip = 0 if self._is_video_file else self._skip_frames
        while self._active and self._capture and self._capture.isOpened():
            ret, frame = self._capture.read()
            if not ret:
                # 视频文件播放完毕，通知停止
                if self._is_video_file:
                    self._active = False
                    self._infer_gen += 1
                break
            with self._frame_lock:
                self._latest_frame = frame

            local_count += 1
            if local_count % (skip + 1) == 0:
                if self._frame_queue is not None:
                    try:
                        self._frame_queue.put((frame.copy(), self._infer_gen), block=False)
                    except Exception:
                        pass  # 队列满时丢弃帧

            # 视频文件按原始帧率读取（摄像头硬件自动限速）
            if self._is_video_file:
                time.sleep(1.0 / self._video_fps)

    # ---- 子进程推理管理 ----
    def _start_worker(self):
        """启动子进程推理"""
        self._frame_queue = self._mp_ctx.Queue(maxsize=2)
        self._result_queue = self._mp_ctx.Queue(maxsize=4)
        self._stop_event = self._mp_ctx.Event()

        self._worker = self._mp_ctx.Process(
            target=inference_worker,
            args=(self._model, self._core_mask, self._param_confidence,
                  self._param_nms, self._frame_queue, self._result_queue,
                  self._stop_event),
            daemon=True,
        )
        self._worker.start()

    def _stop_worker(self):
        """停止子进程推理（先排空结果队列防死锁）"""
        if self._stop_event is not None:
            self._stop_event.set()

        # 先排空结果队列，防止子进程卡在 result_queue.put() 上
        if self._result_queue is not None:
            while True:
                try:
                    self._result_queue.get_nowait()
                except Exception:
                    break

        # 发送退出哨兵
        if self._frame_queue is not None:
            try:
                self._frame_queue.put(None, timeout=0.3)
            except Exception:
                pass

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=1.0)
        self._worker = None
        self._frame_queue = None
        self._result_queue = None
        self._stop_event = None

    def _poll_result(self):
        """定时轮询子进程推理结果"""
        # 检测子进程是否崩溃
        if self._worker is not None and not self._worker.is_alive() and self._active:
            self._active = False
            self._stop()
            self._display.setText(f"摄像头 {self.cam_id} — 推理进程异常退出")
            return
        # 取出所有可用结果
        while True:
            try:
                result = self._result_queue.get_nowait()
            except Exception:
                break
            self._infer_done.emit(result)

    def _display_tick(self):
        """主线程定时回调：显示最新帧（不阻塞、不调度推理）"""
        # 视频文件播放完毕自动停止
        if not self._active:
            self._stop()
            return
        if self._capture is None or not self._capture.isOpened():
            self._stop()
            return

        display_frame = None
        if self._last_annotated is not None:
            display_frame = self._last_annotated
        else:
            with self._frame_lock:
                if self._latest_frame is not None:
                    display_frame = self._latest_frame

        if display_frame is not None:
            self._set_frame(display_frame)

    def _on_infer_done(self, result):
        annotated, info, ms, gen = result
        if gen != self._infer_gen:
            return
        self._last_annotated = annotated
        self._set_frame(annotated)
        if info:
            top = info[0]
            self._save_and_record(annotated, top["label"])

    def _save_and_record(self, frame, label):
        # 健康叶/健康不记入识别记录
        if label in ("健康叶", "健康"):
            return
        now = time.time()
        last_time = self._last_save_time.get(label, 0)
        if now - last_time < self._param_save_interval:
            return
        self._last_save_time[label] = now
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        cv2.imwrite(filepath, frame)
        # 通知父级添加记录
        if hasattr(self, '_on_record'):
            self._on_record(label, filepath, frame)

    def set_frame(self, pixmap):
        """设置视频帧（保持原始比例居中）"""
        if not pixmap or pixmap.isNull():
            return
        scaled = pixmap.scaled(
            self._display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._display.setPixmap(scaled)

    def _set_frame(self, frame_bgr):
        """显示 BGR 帧"""
        if frame_bgr is None:
            return
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self._display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._display.setPixmap(scaled)

    # ---- 图片/视频文件载入（用于打开图片/打开视频） ----
    def load_image(self, file_path):
        frame = cv2.imread(file_path)
        if frame is None:
            return False
        # 单图推理使用主进程检测器（无并发冲突）
        self._detector.switch_model(self._model)
        self._detector.confidence = self._param_confidence
        self._detector.nms_threshold = self._param_nms
        try:
            annotated, info = self._detector.predict_and_draw(frame)
        except Exception:
            annotated = frame
        self._last_annotated = annotated
        self._set_frame(annotated)
        self._status_dot.setStyleSheet("QLabel { background: #007AFF; border-radius: 5px; border: none; }")
        self._display.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 0, 0, 0.30);
                border: 2px solid rgba(0, 122, 255, 0.5);
                border-radius: 14px;
                color: rgba(255, 255, 255, 0.7);
                font-size: {FONT_SIZE_PLACEHOLDER}px;
                font-family: "SF Pro Display";
            }}
        """)
        if info:
            pest_type = info[0]["label"]
            if pest_type not in ("健康叶", "健康"):
                self._on_record(pest_type, file_path, annotated)
        return True

    def load_video(self, file_path):
        self._capture = cv2.VideoCapture(file_path)
        if not self._capture.isOpened():
            return False
        self._is_video_file = True
        self._active = True
        self._latest_frame = None
        self._last_annotated = None
        self._last_save_time.clear()
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._video_fps = fps if fps > 0 else 25

        # 释放主进程模型，启动子进程推理
        self._detector.release()
        self._start_worker()

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._display_timer.start(int(1000 / self._video_fps))
        self._result_poll_timer.start(30)
        self._set_params_enabled(False)

        # 按钮状态同步为"停止"
        self._btn_toggle.blockSignals(True)
        self._btn_toggle.setChecked(True)
        self._btn_toggle.blockSignals(False)
        self._btn_toggle.setText("  停止")
        self._btn_toggle.setIcon(qta.icon('fa5s.stop', color='white'))

        self._status_dot.setStyleSheet("QLabel { background: #FF9500; border-radius: 5px; border: none; }")
        self._display.setStyleSheet(f"""
            QLabel {{
                background: rgba(0, 0, 0, 0.30);
                border: 2px solid rgba(255, 149, 0, 0.5);
                border-radius: 14px;
                color: rgba(255, 255, 255, 0.7);
                font-size: {FONT_SIZE_PLACEHOLDER}px;
                font-family: "SF Pro Display";
            }}
        """)
        return True

    # ---- 截图 ----
    def screenshot(self):
        if self._last_annotated is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.jpg")
            cv2.imwrite(fp, self._last_annotated)
            return fp
        return None

    # ---- 属性 ----
    @property
    def active(self):
        return self._active

    def set_record_callback(self, cb):
        self._on_record = cb

    def cleanup(self):
        self._stop()
        self._stop_worker()
        self._detector.release()


# ============================================================
# 识别页面（双路摄像头）
# ============================================================
class DetectionPage(QWidget):
    status_message = pyqtSignal(str)
    record_added = pyqtSignal(str, str)

    # NPU 核心分配: 摄像头1 用 core 0，摄像头2 用 core 1（2.1.0 runtime 不支持组合掩码）
    NPU_CORE_CAM1 = 1   # NPU_CORE_0
    NPU_CORE_CAM2 = 2   # NPU_CORE_1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._records = []
        self._init_ui()

        # 恢复持久化记录
        for rec in rm_load_records():
            self._restore_record_item(rec.get("type", ""), rec.get("path", ""))

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        layout.addWidget(self._create_button_bar())

        main_row = QHBoxLayout()
        main_row.setSpacing(10)

        # 左侧：双路摄像头左右平铺
        camera_row = QHBoxLayout()
        camera_row.setSpacing(10)
        self._cam1 = _CameraFeed(1, "摄像头 1", CAMERA_DEVICE_1, self.NPU_CORE_CAM1)
        self._cam2 = _CameraFeed(2, "摄像头 2", CAMERA_DEVICE_2, self.NPU_CORE_CAM2)

        def make_cb(cam_id):
            def cb(pest_type, image_path, frame):
                self.add_detection_record(pest_type, image_path, frame, source=f"摄像头{cam_id}")
            return cb
        self._cam1.set_record_callback(make_cb(1))
        self._cam2.set_record_callback(make_cb(2))

        camera_row.addWidget(self._cam1, 1)
        camera_row.addWidget(self._cam2, 1)
        main_row.addLayout(camera_row, 7)

        # 右侧：识别记录
        main_row.addWidget(self._create_record_area(), 2)

        layout.addLayout(main_row, 1)

    # ---- 按钮栏 ----
    def _create_button_bar(self):
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
            }
        """)
        container.setGraphicsEffect(_shadow(20, 0, 6, QColor(0, 0, 0, 50)))

        layout = QHBoxLayout(container)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(14)

        btn_style = """
            QPushButton {
                padding: 10px 18px;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                color: white;
                min-height: 24px;
                font-family: "SF Pro Display";
            }
            QPushButton:hover { opacity: 0.85; }
        """

        btn_image = QPushButton("  打开图片")
        btn_image.setIcon(qta.icon('fa5s.image', color='white'))
        btn_image.setIconSize(QSize(ICON_SIZE_BTN, ICON_SIZE_BTN))
        btn_image.setStyleSheet(btn_style + f"""
            QPushButton {{ background: {_bg('#007AFF', '#5856D6')}; }}
        """)
        btn_image.clicked.connect(self._open_image)
        layout.addWidget(btn_image)

        btn_video = QPushButton("  打开视频")
        btn_video.setIcon(qta.icon('fa5s.video', color='white'))
        btn_video.setIconSize(QSize(ICON_SIZE_BTN, ICON_SIZE_BTN))
        btn_video.setStyleSheet(btn_style + f"""
            QPushButton {{ background: {_bg('#FF2D55', '#FF6B8A')}; }}
        """)
        btn_video.clicked.connect(self._open_video)
        layout.addWidget(btn_video)

        btn_stop_all = QPushButton("  全部停止")
        btn_stop_all.setIcon(qta.icon('fa5s.stop-circle', color='white'))
        btn_stop_all.setIconSize(QSize(ICON_SIZE_BTN, ICON_SIZE_BTN))
        btn_stop_all.setStyleSheet(btn_style + f"""
            QPushButton {{ background: {_bg('#FF9500', '#FFB340')}; }}
        """)
        btn_stop_all.clicked.connect(self._stop_all)
        layout.addWidget(btn_stop_all)

        btn_screenshot = QPushButton("  截图")
        btn_screenshot.setIcon(qta.icon('fa5s.camera', color='white'))
        btn_screenshot.setIconSize(QSize(ICON_SIZE_BTN, ICON_SIZE_BTN))
        btn_screenshot.setStyleSheet(btn_style + f"""
            QPushButton {{ background: {_bg('#34C759', '#30D158')}; }}
        """)
        btn_screenshot.clicked.connect(self._screenshot)
        layout.addWidget(btn_screenshot)

        layout.addStretch()

        return container

    # ---- 记录区 ----
    def _create_record_area(self):
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 24px;
            }
        """)
        container.setGraphicsEffect(_shadow(40, 0, 15, QColor(0, 0, 0, 60)))

        layout = QVBoxLayout(container)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title_label = QLabel("识别记录")
        title_label.setFont(QFont("SF Pro Display", 13, QFont.Bold))
        title_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                padding: 6px 12px;
                background: {_bg('#007AFF', '#5856D6')};
                border-radius: 8px;
            }}
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFixedHeight(45)
        layout.addWidget(title_label)

        self._record_scroll = QScrollArea()
        self._record_scroll.setWidgetResizable(True)
        self._record_scroll.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #f0f0f0; width: 8px; border-radius: 4px; }
            QScrollBar::handle:vertical { background: #cccccc; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #aaaaaa; }
        """)
        self._record_container = QWidget()
        self._record_layout = QVBoxLayout(self._record_container)
        self._record_layout.setSpacing(10)
        self._record_layout.setAlignment(Qt.AlignTop)
        self._record_layout.addStretch()
        self._record_scroll.setWidget(self._record_container)
        layout.addWidget(self._record_scroll, 1)

        btn_clear = QPushButton("  清空记录")
        btn_clear.setIcon(qta.icon('fa5s.trash-alt', color='#666666'))
        btn_clear.setIconSize(QSize(ICON_SIZE_CLEAR_BTN, ICON_SIZE_CLEAR_BTN))
        btn_clear.setStyleSheet(f"""
            QPushButton {{
                padding: 12px;
                border: none;
                border-radius: 12px;
                font-size: {FONT_SIZE_BTN}px;
                font-weight: bold;
                color: #666666;
                background: #f0f0f0;
                font-family: "SF Pro Display";
            }}
            QPushButton:hover {{ background: #e0e0e0; color: #333333; }}
        """)
        btn_clear.clicked.connect(self._clear_records)
        layout.addWidget(btn_clear)
        return container

    # ---- 按钮操作 ----
    def _open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.tif)"
        )
        if not file_path:
            return
        self._stop_all()
        ok = self._cam1.load_image(file_path)
        if not ok:
            self.status_message.emit("无法读取图片")
        else:
            self.status_message.emit(f"已打开图片: {os.path.basename(file_path)}")

    def _open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if not file_path:
            return
        self._stop_all()
        ok = self._cam1.load_video(file_path)
        if not ok:
            self.status_message.emit("无法打开视频")
        else:
            self.status_message.emit(f"正在播放: {os.path.basename(file_path)}")

    def _stop_all(self):
        self._cam1._stop()
        self._cam2._stop()
        self.status_message.emit("全部摄像头已停止")

    def _screenshot(self):
        fp1 = self._cam1.screenshot()
        fp2 = self._cam2.screenshot()
        if fp1:
            self.status_message.emit(f"截图已保存: {os.path.basename(fp1)}")
        if fp2:
            self.status_message.emit(f"截图已保存: {os.path.basename(fp2)}")
        if not fp1 and not fp2:
            self.status_message.emit("没有可截图的内容")

    def _clear_records(self):
        while self._record_layout.count() > 1:
            item = self._record_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._records.clear()
        rm_clear_records()
        self.status_message.emit("记录已清空")

    # ---- 记录管理 ----
    def add_detection_record(self, pest_type, image_path, frame=None, source=""):
        timestamp = datetime.now().strftime("%H:%M:%S")

        item = RecordItem(timestamp, f"[{source}] {pest_type}" if source else pest_type, image_path)
        item.clicked.connect(lambda: self._jump_to_record(image_path))

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            q_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            item.set_thumbnail(pixmap)
        elif os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            item.set_thumbnail(pixmap)

        self._records.append({"time": timestamp, "type": pest_type, "path": image_path})
        rm_add_record(pest_type, image_path)
        self._record_layout.insertWidget(self._record_layout.count() - 1, item)
        self.status_message.emit(f"识别记录: [{source}] {pest_type} ({timestamp})" if source else f"识别记录: {pest_type} ({timestamp})")
        self.record_added.emit(pest_type, image_path)

    def _restore_record_item(self, pest_type, image_path):
        if not pest_type or not image_path:
            return
        basename = os.path.basename(image_path)
        parts = basename.replace(".jpg", "").split("_", 1)
        timestamp = parts[1] if len(parts) > 1 else pest_type

        item = RecordItem(timestamp, pest_type, image_path)
        item.clicked.connect(lambda checked=False, p=image_path: self._jump_to_record(p))
        if os.path.exists(image_path):
            item.set_thumbnail(QPixmap(image_path))
        self._records.append({"time": timestamp, "type": pest_type, "path": image_path})
        self._record_layout.insertWidget(self._record_layout.count() - 1, item)

    def _jump_to_record(self, image_path):
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "文件不存在或已被删除")
            return
        self._stop_all()
        frame = cv2.imread(image_path)
        if frame is not None:
            pixmap = QPixmap(image_path)
            self._cam1.set_frame(pixmap)
            self.status_message.emit(f"跳转查看: {os.path.basename(image_path)}")

    # ---- 生命周期 ----
    def on_leave(self):
        """离开页面：停止所有摄像头和子进程推理"""
        self._stop_all()

    def on_enter(self):
        """进入页面：无需额外操作"""
        pass

    def cleanup(self):
        """程序退出：完全清理资源"""
        self._stop_all()
        self._cam1.cleanup()
        self._cam2.cleanup()
        rm_clear_records()
        self.status_message.emit("已停止")
