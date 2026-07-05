import threading
import serial
import pynmea2


# ---- 配置参数 ----
SERIAL_PORT = '/dev/ttyS9'
BAUD_RATE = 38400
TIMEOUT = 1


class GPSService:
    """GPS服务：读取经纬度和海拔"""

    def __init__(self):
        self.latitude = None
        self.longitude = None
        self.altitude = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        """启动GPS读取线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止GPS读取"""
        self._running = False

    def _read_loop(self):
        """持续读取GPS数据"""
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
            print(f"GPS: 成功打开串口 {SERIAL_PORT} @ {BAUD_RATE}bps")

            while self._running:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if not line.startswith('$'):
                    continue

                if line.startswith('$GNGGA') or line.startswith('$GPGGA'):
                    try:
                        msg = pynmea2.parse(line)
                        if msg.gps_qual > 0:
                            with self._lock:
                                self.latitude = msg.latitude
                                self.longitude = msg.longitude
                                self.altitude = msg.altitude
                                print(f"GPS: 纬度={self.latitude:.6f} 经度={self.longitude:.6f} 海拔={self.altitude}m")
                    except pynmea2.ParseError:
                        pass

        except serial.SerialException as e:
            print(f"GPS串口错误: {e}")
        finally:
            if 'ser' in locals() and ser.is_open:
                ser.close()

    def get_location(self):
        """获取位置信息，返回 (纬度, 经度, 海拔) 或 None"""
        with self._lock:
            if self.latitude is not None:
                return (self.latitude, self.longitude, self.altitude)
        return None


# 全局GPS服务实例
gps_service = GPSService()


if __name__ == "__main__":
    gps_service.start()
    try:
        while True:
            print(gps_service.get_location())
            import time
            time.sleep(2)
    except KeyboardInterrupt:
        gps_service.stop()
