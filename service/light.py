import os
import threading
import time
import statistics
from pathlib import Path

from config import (
    LIGHT_SENSOR_ADC_DEVICE,
    LIGHT_SENSOR_ADC_CHANNEL,
    LIGHT_SENSOR_VREF,
    LIGHT_SENSOR_ADC_BITS,
    LIGHT_SENSOR_SAMPLES,
    LIGHT_SENSOR_SAMPLE_DELAY,
    LIGHT_SENSOR_INTERVAL,
    LIGHT_SENSOR_R_TOP,
    LIGHT_SENSOR_R_BOTTOM,
    LIGHT_SENSOR_CAL_VOLTAGE,
    LIGHT_SENSOR_DARK_VOLTAGE,
)


def _read_number(path):
    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


class LightSensorService:
    """光照强度：通过 RK3588 ADC sysfs 读取太阳能面板电压"""

    def __init__(self):
        self._callbacks = []
        self._lock = threading.Lock()
        self._last_data = {}
        self._running = False
        self._thread = None

        self._device = LIGHT_SENSOR_ADC_DEVICE
        self._channel = LIGHT_SENSOR_ADC_CHANNEL
        self._vref = LIGHT_SENSOR_VREF
        self._raw_max = (1 << LIGHT_SENSOR_ADC_BITS) - 1
        self._samples = LIGHT_SENSOR_SAMPLES
        self._sample_delay = LIGHT_SENSOR_SAMPLE_DELAY
        self._interval = LIGHT_SENSOR_INTERVAL
        self._divider_ratio = (LIGHT_SENSOR_R_TOP + LIGHT_SENSOR_R_BOTTOM) / LIGHT_SENSOR_R_BOTTOM
        self._cal_voltage = LIGHT_SENSOR_CAL_VOLTAGE
        self._dark_voltage = LIGHT_SENSOR_DARK_VOLTAGE

        self._raw_path = Path(self._device) / f"in_voltage{self._channel}_raw"
        self._scale_path = Path(self._device) / "in_voltage_scale"

    def add_callback(self, cb):
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb):
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)

    def get_last_data(self):
        with self._lock:
            return dict(self._last_data)

    def is_available(self):
        return self._raw_path.exists()

    def _read_voltage(self):
        raw = int(_read_number(self._raw_path))
        if self._scale_path.exists():
            adc_voltage = raw * _read_number(self._scale_path) / 1000.0
        else:
            adc_voltage = raw * self._vref / self._raw_max
        return raw, adc_voltage

    def _sample_average(self):
        raws = []
        volts = []
        for i in range(self._samples):
            raw, adc_voltage = self._read_voltage()
            raws.append(raw)
            volts.append(adc_voltage)
            if i != self._samples - 1:
                time.sleep(self._sample_delay)
        return round(statistics.fmean(raws)), statistics.fmean(volts)

    def _estimate_irradiance(self, panel_voltage):
        if self._cal_voltage <= self._dark_voltage:
            return 0.0
        watts = (panel_voltage - self._dark_voltage) * 1000.0 / (self._cal_voltage - self._dark_voltage)
        return max(0.0, min(watts, 2000.0))

    def _read_once(self):
        if not self._raw_path.exists():
            return None
        try:
            raw, adc_voltage = self._sample_average()
            panel_voltage = adc_voltage * self._divider_ratio
            irradiance = self._estimate_irradiance(panel_voltage)
            return {
                "raw": raw,
                "adc_voltage": round(adc_voltage, 4),
                "panel_voltage": round(panel_voltage, 3),
                "irradiance": round(irradiance, 1),
            }
        except Exception:
            return None

    def _loop(self):
        while self._running:
            data = self._read_once()
            if data:
                with self._lock:
                    self._last_data = data
                    callbacks = list(self._callbacks)
                for cb in callbacks:
                    try:
                        cb(data)
                    except Exception:
                        pass
            time.sleep(self._interval)

    def start(self):
        if self._running:
            return
        if not self.is_available():
            raise FileNotFoundError(f"ADC 路径不存在: {self._raw_path}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


light_sensor_service = LightSensorService()
