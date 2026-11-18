from .config import make_qos
# 한 칸 버퍼와 재연결 스레드로 최신 카메라 프레임을 제공한다.
import logging
import json
import platform
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np


class JetsonFileCapture:
    """Hardware decode with OpenCV metadata and a software fallback for seeking."""
    def __init__(self, uri):
        self.uri = str(uri)
        self.capture = cv2.VideoCapture(self.uri)
        self.hardware = False
        self.pending = None
        self.position = 0
        self.orientation = 0
        if not self.capture.isOpened():
            return
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.orientation = int(self.capture.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
        if self.orientation not in (0, 90, 180, 270):
            self.orientation = 0
        fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode('ascii', 'replace').lower()
        codec = {'hvc1': 'h265', 'hev1': 'h265', 'avc1': 'h264', 'h264': 'h264'}.get(fourcc)
        # Some Jetson installations expose the plugin without the decoder device.
        if not codec or not Path('/dev/nvhost-nvdec').exists():
            logging.info('Jetson hardware decoder unavailable; using OpenCV file capture')
            return
        pipeline = (f'filesrc location={json.dumps(self.uri)} ! qtdemux name=demux '
                    f'demux.video_0 ! queue ! {codec}parse ! nvv4l2decoder ! '
                    'nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! '
                    'video/x-raw,format=BGR ! appsink sync=false')
        candidate = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        ok, image = candidate.read() if candidate.isOpened() else (False, None)
        if not ok:
            candidate.release()
            logging.warning('Jetson hardware decoder failed; using OpenCV file capture')
            return
        self.capture.release()
        self.capture, self.hardware, self.pending = candidate, True, image
        logging.info('Jetson hardware file decoder enabled')

    def isOpened(self):
        return self.capture.isOpened()

    def read(self):
        if self.pending is not None:
            image, self.pending = self.pending, None
            ok = True
        else:
            ok, image = self.capture.read()
        if ok:
            self.position += 1
            if self.hardware and self.orientation:
                image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[self.orientation])
        return ok, image

    def grab(self):
        if self.pending is not None:
            self.pending = None
            ok = True
        else:
            ok = self.capture.grab()
        if ok:
            self.position += 1
        return ok

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.count
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self.capture.get(prop)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES and self.hardware:
            # GStreamer frame seeking depends on the container; OpenCV's normal
            # capture retains the existing seek and orientation semantics.
            replacement = cv2.VideoCapture(self.uri)
            if not replacement.isOpened() or not replacement.set(prop, value):
                replacement.release()
                return False
            self.capture.release()
            self.capture, self.hardware, self.pending = replacement, False, None
            self.position = int(value)
            return True
        ok = self.capture.set(prop, value)
        if ok and prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return ok

    def release(self):
        self.capture.release()


class Source:
    # 디렉터리 목록은 시작 시 한 번만 읽는다.
    def __init__(self, config):
        self.c = config
        self.stop = threading.Event()
        self.cv = threading.Condition()
        self.slot, self.errors = None, deque()
        self.done, self.index, self.dropped = False, 0, 0
        self.eof = False
        self.last_received = time.perf_counter()
        self.capture, self.node, self.context, self.executor = None, None, None, None
        self.paths = sorted(p for p in Path(config['uri']).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}) if config['type'] == 'dir' else []
        if config['limit']:
            self.paths = self.paths[:config['limit']]
        self.speed = float(config['speed'])
        self.paused = False
        self.skip_fraction = 0.0
        self.seek_target = None
        self.seek_generation = 0
        self.fps = float(config['fps'])
        self.total_frames = 0
        self.thread = threading.Thread(target=self._loop, name='vision-input', daemon=True)

    # 소비 대기 또는 최신 프레임 교체를 수행한다.
    def _put(self, image, started, generation=None, header=None):
        with self.cv:
            while self.slot is not None and not self.c['latest_only'] and self.c['type'] != 'ros2' and not self.stop.is_set():
                self.cv.wait(.1)
            if self.stop.is_set() or (generation is not None and generation != self.seek_generation):
                return
            self.index += 1
            self.last_received = time.perf_counter()
            self.dropped += self.slot is not None
            timestamp = header['stamp']['sec'] + header['stamp']['nanosec'] / 1e9 if header else time.time()
            self.slot = (self.index, image, timestamp, time.perf_counter(), (time.perf_counter()-started)*1000, header)
            self.cv.notify_all()

    # ROS 이미지의 인코딩·stride·endianness를 검사해 BGR로 복사한다.
    def _ros_callback(self, msg):
        started = time.perf_counter()
        try:
            channels = {'bgr8':3, 'rgb8':3, 'mono8':1, 'bgra8':4, 'rgba8':4}[msg.encoding]
            if msg.height <= 0 or msg.width <= 0 or msg.step < msg.width * channels:
                raise ValueError('invalid image dimensions/stride')
            raw = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
            a = raw[:, :msg.width*channels].reshape(msg.height, msg.width, channels).copy()
            code = {'rgb8':cv2.COLOR_RGB2BGR, 'mono8':cv2.COLOR_GRAY2BGR,
                    'bgra8':cv2.COLOR_BGRA2BGR, 'rgba8':cv2.COLOR_RGBA2BGR}.get(msg.encoding)
            header = dict(stamp=dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec),
                          frame_id=msg.header.frame_id)
            # Compare against the subscriber's ROS clock (also supports /clock).
            stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            age_s = (self.node.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if stamp_ns == 0 or age_s < 0 or age_s > self.c['timeout_s']:
                raise ValueError('invalid_or_stale_image_stamp')
            header['age_at_receive_ms'] = age_s * 1000
            self._put(cv2.cvtColor(a, code) if code is not None else a, started, header=header)
        except Exception as exc:
            self._error(exc)

    # ROS 및 OpenCV 입력을 종류에 맞게 연다.
    def _open(self):
        c, kind = self.c, self.c['type']
        if kind == 'dir':
            return
        if kind == 'ros2':
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import Image
            self.context = Context()
            rclpy.init(context=self.context)
            self.node = rclpy.create_node('vision_input', context=self.context)
            qos = make_qos(c['ros2_qos'])
            self.node.create_subscription(Image, c['topic'], self._ros_callback, qos)
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            return
        uri = c['uri']
        api = cv2.CAP_ANY
        if kind == 'csi':
            uri = (f"nvarguscamerasrc sensor-id={c['csi_sensor']} ! video/x-raw(memory:NVMM),width={c['width']},height={c['height']},framerate={int(c['fps'])}/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers={c['buffer_size']} sync=false")
            api = cv2.CAP_GSTREAMER
        elif kind == 'v4l2':
            api = cv2.CAP_V4L2
        elif kind not in ('rtsp', 'file'):
            raise ValueError('unknown source: ' + kind)
        if kind == 'file' and platform.machine() == 'aarch64':
            self.capture = JetsonFileCapture(uri)
            if not self.capture.isOpened():
                raise OSError('cannot open ' + kind)
            return
        self.capture = cv2.VideoCapture()
        if kind == 'rtsp':
            self.capture.open(uri, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(c['timeout_s']*1000), cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(c['timeout_s']*1000)])
        else:
            self.capture.open(uri, api)
        if not self.capture.isOpened():
            raise OSError('cannot open ' + kind)
        if kind == 'v4l2':
            for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH,c['width']), (cv2.CAP_PROP_FRAME_HEIGHT,c['height']), (cv2.CAP_PROP_FPS,c['fps']), (cv2.CAP_PROP_BUFFERSIZE,c['buffer_size'])):
                self.capture.set(prop, value)

    # 오류를 프레임과 별도 보관해 정지 이벤트가 덮어써지지 않게 한다.
    def _error(self, exc):
        logging.error('input disconnected: %s', exc)
        with self.cv:
            self.errors.clear()
            self.errors.append(str(exc))
            self.slot = None
            self.cv.notify_all()

    # 입력을 해제하고 ROS 컨텍스트를 종료한다.
    def _release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.context is not None:
            self.context.try_shutdown()
            self.context = None

    # 유한 입력의 배속과 일시정지를 스레드 간에 공유한다.
    def controls(self, speed, paused):
        with self.cv:
            self.speed, self.paused = speed, paused
            self.cv.notify_all()

    # 표시 중인 프레임을 기준으로 탐색을 요청하고 이전 버퍼를 비운다.
    def seek(self, frame, seconds):
        if self.c['type'] not in ('file', 'dir'):
            logging.warning('seek ignored for live input')
            return False
        with self.cv:
            self.eof = False
            self.seek_target = (frame, seconds)
            self.seek_generation += 1
            self.slot = None
            self.cv.notify_all()
        return True

    # 재생 시각까지 기다리고 늦어진 입력 프레임 수를 계산한다.
    def _pace(self, deadline, fps):
        with self.cv:
            while not self.stop.is_set():
                if self.seek_target is not None:
                    return time.perf_counter(), 0
                if self.paused:
                    self.cv.wait(.1)
                    deadline = time.perf_counter()
                    continue
                now = time.perf_counter()
                if now >= deadline:
                    interval = 1 / (fps * min(self.speed, 1.0))
                    late = int((now - deadline) / interval)
                    self.skip_fraction += max(self.speed - 1.0, 0.0)
                    extra = int(self.skip_fraction)
                    self.skip_fraction -= extra
                    skipped = int(late * max(self.speed, 1.0)) + extra
                    return deadline + (late + 1) * interval, skipped
                self.cv.wait(min(deadline - now, .1))
        return deadline, 0

    # EOF에서도 탐색 요청을 기다려 마지막 프레임에서 되돌아갈 수 있다.
    def _wait_at_eof(self):
        with self.cv:
            self.eof = True
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.seek_target is not None or self.stop.is_set())
            self.eof = False

    # 유한 소스 EOF는 탐색을 기다리고 읽기 실패는 재시도한다.
    def _loop(self):
        delay, pos = self.c['backoff_s'], 0
        try:
            while not self.stop.is_set():
                try:
                    self._open()
                    logging.info('input connected source=%s position=%s', self.c['type'], pos)
                    fps = self.c['fps']
                    if self.c['type'] == 'file':
                        measured = self.capture.get(cv2.CAP_PROP_FPS)
                        if np.isfinite(measured) and measured > 0:
                            fps = measured
                    self.fps = fps
                    self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) if self.c['type'] == 'file' else len(self.paths)
                    deadline = time.perf_counter()
                    while not self.stop.is_set():
                        if self.c['type'] in ('file', 'dir'):
                            deadline, skip = self._pace(deadline, fps)
                            if self.stop.is_set():
                                break
                            with self.cv:
                                target, self.seek_target = self.seek_target, None
                                generation = self.seek_generation
                            if target is not None:
                                target = max(0, target[0] - 1 + round(target[1] * fps))
                                total = len(self.paths) if self.c['type'] == 'dir' else int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                                target = min(target, max(0, total - 1)) if total > 0 else target
                                if self.c['type'] == 'dir':
                                    pos = target
                                else:
                                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                                self.skip_fraction, skip = 0.0, 0
                                deadline = time.perf_counter()
                            if self.c['type'] == 'dir':
                                pos = min(pos + skip, len(self.paths))
                            else:
                                # Bound each batch so a slow decoder cannot spend
                                # seconds catching up without delivering a frame.
                                skip = min(skip, max(1, round(fps / 4)))
                                skip_start = time.perf_counter()
                                skipped = 0
                                for _ in range(skip):
                                    if self.stop.is_set() or self.seek_target is not None or not self.capture.grab():
                                        break
                                    skipped += 1
                                if skipped and self.index % 30 == 0:
                                    logging.info('source skip=%d grab_ms=%.1f', skipped, (time.perf_counter()-skip_start)*1000)
                                skip = skipped
                            self.dropped += skip
                        started = time.perf_counter()
                        if self.c['type'] == 'ros2':
                            self.executor.spin_once(timeout_sec=.1)
                            continue
                        if self.c['type'] == 'dir':
                            if pos >= len(self.paths):
                                self._wait_at_eof()
                                continue
                            image = cv2.imread(str(self.paths[pos]))
                            if image is None:
                                raise OSError('unreadable frame ' + str(self.paths[pos]))
                            pos += 1
                        else:
                            ok, image = self.capture.read()
                            if not ok:
                                if self.c['type'] == 'file' and self.capture.get(cv2.CAP_PROP_POS_FRAMES) >= self.capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
                                    self._wait_at_eof()
                                    continue
                                raise OSError('capture read failed')
                        if self.c['type'] in ('file', 'dir'):
                            with self.cv:
                                if generation != self.seek_generation:
                                    continue
                                self.index = (pos if self.c['type'] == 'dir' else int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                        self._put(image, started, generation if self.c['type'] in ('file', 'dir') else None)
                        delay = self.c['backoff_s']
                except Exception as exc:
                    self._error(exc)
                finally:
                    self._release()
                logging.info('input reconnect backoff=%.3fs', delay)
                if self.stop.wait(delay):
                    break
                delay = min(delay*2, self.c['backoff_max_s'])
        finally:
            with self.cv:
                self.done = True
                self.cv.notify_all()

    # 타임아웃·EOF·프레임을 구분해 반환한다.
    def get(self, timeout=None):
        with self.cv:
            wait_s = self.c['timeout_s'] if timeout is None else timeout
            if self.c['type'] == 'ros2':
                wait_s = min(wait_s, max(0., self.c['timeout_s'] - (time.perf_counter() - self.last_received)))
            self.cv.wait_for(lambda: self.errors or self.slot is not None or self.eof or self.done or self.stop.is_set(), wait_s)
            if timeout is not None and not (self.errors or self.slot is not None or self.eof or self.done or self.stop.is_set()):
                if self.c['type'] == 'ros2' and time.perf_counter() - self.last_received >= self.c['timeout_s']:
                    self.last_received = time.perf_counter()
                    return 'error', 'input_timeout'
                return 'waiting', None
            if self.errors:
                return 'error', self.errors.popleft()
            if self.slot is not None:
                item, self.slot = self.slot, None
                self.cv.notify_all()
                return 'frame', item
            if self.eof or self.done or self.stop.is_set():
                return 'eof', None
            self.last_received = time.perf_counter()
            return 'error', 'input_timeout'

    # 종료 신호로 대기 중인 생산자를 깨운다.
    def close(self):
        self.stop.set()
        with self.cv:
            self.cv.notify_all()
        self.thread.join(timeout=None if self.c['type'] == 'file' else self.c['timeout_s']+1)
