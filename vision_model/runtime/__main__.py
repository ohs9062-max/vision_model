# 입력부터 안전 판단·출력까지 설정 기반 실행 루프를 연결한다.
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging
import json
import os
import queue
import sys
import threading
import select
from pathlib import Path
import signal
import time
import cv2
import numpy as np
from .backends import Backend
from .stabilizer import create_processors
from .sources import Source
from .sinks import Sinks
from .config import validate_runtime, load_config


# 키 입력을 회전·배속·일시정지 상태로 변환한다.
def handle_key(key, rotate, speed, paused, finite=True):
    steps = (.25, .5, 1., 2., 4.)
    if key in (ord('q'), 27):
        return rotate, speed, paused, False
    if key == ord('r'):
        rotate = (rotate + 90) % 360
    elif finite and key in (ord('+'), ord('='), ord(']')):
        speed = next((v for v in steps if v > speed), steps[-1])
    elif finite and key in (ord('-'), ord('[')):
        speed = next((v for v in reversed(steps) if v < speed), steps[0])
    elif finite and key == ord(' '):
        paused = not paused
    return rotate, speed, paused, True


# 백엔드별 화살표 코드를 탐색 시간으로 변환한다.
def seek_seconds(key, intervals):
    keys = {65361: -intervals[0], 65363: intervals[0], 65362: intervals[1], 65364: -intervals[1],
            2424832: -intervals[0], 2555904: intervals[0], 2490368: intervals[1], 2621440: -intervals[1],
            81: -intervals[0], 83: intervals[0], 82: intervals[1], 84: -intervals[1]}
    return keys.get(key)


# 종료·EOF·입력 오류를 구분하고 이동평균 계측을 출력한다.
def run(config, progress=None, control=False):
    validate_runtime(config)
    ros_mode = config['source']['type'] == 'ros2' or config['sinks']['ros2']['enabled']
    processor_config = dict(config, decision=dict(config['decision'], require_validated=ros_mode or config['decision']['require_validated']))
    stabilizer, decision = create_processors(processor_config)
    sinks = Sinks(config['sinks'])
    source, backend = None, None
    stopped, processed, emitted = False, 0, 0
    history = deque(maxlen=config['monitor']['window'])
    ticks = deque(maxlen=config['monitor']['window'])
    old_handlers = {}
    rotate = config['source']['rotate']
    speed = config['source']['speed']
    if rotate not in (0, 90, 180, 270) or not .25 <= speed <= 4 or not np.isfinite(speed):
        raise ValueError('source.rotate must be 0/90/180/270 and source.speed must be 0.25..4')
    finite = config['source']['type'] in ('file', 'dir')
    if not finite:
        logging.warning('playback speed ignored for live input')
        speed = 1.0
    paused = False
    commands = queue.Queue()
    last_report = 0.0
    last_success_timestamp = None
    current_header = None
    frame = 0
    seek_generation = 0
    decision_lock = threading.Lock()
    executor = None
    prefetched = None
    # 표준 입력을 별도 스레드에서 받아 재생 루프를 막지 않는다.
    command_stop = threading.Event()
    command_thread = None
    def read_commands():
        pending = b''
        while not command_stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], .1)
            if ready:
                chunk = os.read(sys.stdin.fileno(), 4096)
                if not chunk:
                    break
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    commands.put(line.decode('utf-8', errors='replace').strip())
    if control:
        command_thread = threading.Thread(target=read_commands, name='vision-control')
        command_thread.start()

    # 버튼 명령과 키 입력을 같은 탐색·상태 변경 함수로 처리한다.
    def apply_key(key, intervals=None):
        nonlocal rotate, speed, paused, stopped, seek_generation
        if sys.platform.startswith("linux") and 0 <= key <= 0x1fffff:
            key &= 0xffff  # GTK includes X11 modifier flags above the keysym.
        delta = seek_seconds(key, intervals or config['sinks']['display']['seek_seconds'])
        sought = delta is not None and source.seek(frame, delta)
        if sought or key == ord('r') or key == ord(' '):
            with decision_lock:
                seek_generation += 1
                if sought:
                    decision.pending, decision.count = None, 0
                    decision.nonstop_count = 0
                    stabilizer.reset()
        sinks.handle_window_key(key)
        rotate, speed, paused, keep = handle_key(key, rotate, speed, paused, finite)
        source.controls(speed, paused)
        if not keep:
            stopped = True
            if control:
                print(json.dumps(dict(event='stopping', time=time.monotonic())), flush=True)
        if key >= 0:
            logging.info("control key=%s seek=%s sought=%s frame=%s paused=%s speed=%s rotate=%s fullscreen=%s",
                         key, delta, sought, frame, paused, speed, rotate, sinks.fullscreen)
        return sought

    # 표시된 프레임과 입력 길이를 0.5초마다 제어 창에 전달한다.
    def service_control():
        nonlocal last_report
        if not control:
            return False
        sought = False
        keys = {'back30': 84, 'back5': 81, 'pause': ord(' '), 'forward5': 83,
                'forward30': 82, 'rotate': ord('r'), 'slower': ord('-'), 'faster': ord('+'),
                'fullscreen': ord('f'), 'reset_window': ord('1'), 'stop': ord('q')}
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break
            if command in keys:
                sought = apply_key(keys[command], [5, 30]) or sought
        now = time.monotonic()
        if now - last_report >= .5:
            with source.cv:
                fps = source.fps
                total = len(source.paths) if source.c['type'] == 'dir' else source.total_frames
            print(json.dumps(dict(position=frame / fps if fps else 0, duration=total / fps if fps else 0,
                                  speed=speed, rotate=rotate, paused=paused, fullscreen=sinks.fullscreen)), flush=True)
            last_report = now
        return sought

    # 추론, 후처리, 판단을 한 작업으로 묶어 표시와 동시에 진행한다.
    def infer_and_decide(image, generation, input_ms, image_rotate):
        result, stages = backend(image)
        result['rotate'] = image_rotate
        if result['mask'].dtype != np.uint8 or result['mask'].shape != image.shape[:2] or not np.isin(result['mask'], (0, 1, 2)).all():
            raise ValueError('invalid segmentation mask')
        if not np.isfinite(result['mask']).all() or any(not np.isfinite(b).all() or not np.isfinite(s) for b, s, c in result['boxes']):
            raise FloatingPointError('nonfinite predictor result')
        with decision_lock:
            if generation != seek_generation:
                return result, stages, None
            t = time.perf_counter()
            before = decision.evaluate(result) if ros_mode else None
            if before is not None:
                before.update(box_count=len(result['boxes']),
                              detected_classes=sorted({decision.c['box_classes'][int(b[2])] for b in result['boxes']}))
            result['boxes'] = stabilizer.update(result['boxes'])
            state = decision(result)
            if before is not None:
                state['pre_stabilizer'] = before
            stages.update(input=input_ms, decision=(time.perf_counter()-t)*1000)
        return result, stages, state

    # 외부 종료 신호는 입력 대기를 깨우고 안전 정지를 요청한다.
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if source:
            source.stop.set()
            with source.cv:
                source.cv.notify_all()

    # 모든 이벤트를 같은 JSON 스키마로 출력한다.
    def emit(state, frame=0, timestamp=None, stages=None, image=None, result=None, event='frame', age_ms=0):
        nonlocal emitted, last_success_timestamp
        stages = stages or dict(input=0., inference=0., postprocess=0., decision=0.)
        stages['total'] = sum(stages.values())
        if event == 'frame':
            history.append(stages.copy())
            ticks.append(time.perf_counter())
        means = {k: float(np.mean([v[k] for v in history])) if history else 0. for k in stages}
        fps = (len(ticks)-1)/(ticks[-1]-ticks[0]) if len(ticks)>1 and ticks[-1]>ticks[0] else 0.
        emitted += 1
        if event == 'frame':
            last_success_timestamp = timestamp
        packet = dict(timestamp=time.time() if timestamp is None else timestamp, frame=frame, event=event, sequence=emitted, **state,
                      boxes=result['boxes'] if result else [], latency_ms=stages, moving_latency_ms=means,
                      frame_age_ms=age_ms, fps=fps, dropped_frames=source.dropped if source else 0, rotate=rotate, speed=speed)
        packet.update(header=current_header, last_success_timestamp=last_success_timestamp,
                      inference_status='ok' if event == 'frame' else event,
                      policy_validated=config['decision'].get('policy_validated') is True,
                      decision_valid=event == 'frame' and (not ros_mode or config['decision'].get('policy_validated') is True),
                      detected_classes=sorted({config['decision']['box_classes'][int(b[2])] for b in packet['boxes']}))
        sink_start = time.perf_counter()
        keep = sinks.emit(packet, image, result)
        if event == 'frame' and emitted % config['monitor']['log_every'] == 0:
            logging.info('display_emit_ms=%.1f', (time.perf_counter()-sink_start)*1000)
        if emitted % config['monitor']['log_every'] == 0 or event != 'frame':
            logging.info('event=%s frame=%s decision=%s fps=%.2f latency_ms=%s', event, frame, state['decision'], fps, means)
        if progress and event == 'frame':
            progress(processed, packet)
        return keep

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, stop)
        if ros_mode:
            emit(decision.fail('startup'), event='startup')
        try:
            backend = Backend(config['backend'])
            source = Source(config['source'])
        except Exception as exc:
            emit(decision.fail('initialization_failed: ' + str(exc)), event='error')
            raise
        source.thread.start()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-inference")
        while not stopped:
            service_control()
            if sinks.display and sinks.window_open:
                while True:
                    key = cv2.waitKeyEx(1)
                    if key < 0:
                        break
                    apply_key(key)
            if stopped:
                break
            wait_start = time.perf_counter()
            prepared = None
            if prefetched is None:
                kind, item = source.get(timeout=.01 if (control or sinks.display) else None)
            else:
                kind, item, prepared = prefetched
                prefetched = None
                if prepared is not None and prepared[1] != seek_generation:
                    continue
            if kind == 'frame' and processed % config['monitor']['log_every'] == 0:
                logging.info('source_wait_ms=%.1f', (time.perf_counter()-wait_start)*1000)
            if kind == 'waiting':
                continue
            if stopped:
                break
            if kind == 'eof':
                # 마지막 화면에서 탐색·종료 키를 받으며 창이 없으면 종료한다.
                if (not sinks.display and not control) or not processed:
                    break
                if sinks.display:
                    apply_key(cv2.waitKeyEx(50))
                else:
                    time.sleep(.05)
                continue
            if kind == 'error':
                stabilizer.reset()
                current_header = None
                emit(decision.fail('input_failed: ' + str(item)), event='error')
                continue
            frame, image, timestamp, captured, input_ms, current_header = item
            if prepared is None and rotate:
                image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
            age = (time.perf_counter()-captured)*1000 + (current_header or {}).get('age_at_receive_ms', 0)
            if (ros_mode or config['source']['latest_only']) and age > config['source']['timeout_s']*1000:
                stabilizer.reset()
                emit(decision.fail('stale_frame'), frame, timestamp, event='error', age_ms=age)
                continue
            try:
                generation = seek_generation if prepared is None else prepared[1]
                future = executor.submit(infer_and_decide, image, generation, input_ms, rotate) if prepared is None else prepared[0]
                expired = False
                while True:
                    try:
                        result, stages, state = future.result(timeout=.01)
                        break
                    except TimeoutError:
                        if ros_mode and not expired:
                            with source.cv:
                                input_error = source.errors.popleft() if source.errors else None
                            if input_error is not None:
                                with decision_lock:
                                    seek_generation += 1
                                    stabilizer.reset()
                                    emit(decision.fail('input_failed: ' + input_error), frame, timestamp, event='error')
                                expired = True
                        elapsed_age = (time.perf_counter()-captured)*1000 + (current_header or {}).get('age_at_receive_ms', 0)
                        if ros_mode and not expired and elapsed_age > config['source']['timeout_s']*1000:
                            with decision_lock:
                                seek_generation += 1
                                stabilizer.reset()
                                emit(decision.fail('inference_timeout'), frame, timestamp, event='error', age_ms=elapsed_age)
                            expired = True
                        if stopped:
                            break
                        service_control()
                        if sinks.display and sinks.window_open:
                            while True:
                                key = cv2.waitKeyEx(1)
                                if key < 0:
                                    break
                                apply_key(key)
                if stopped:
                    break
                if generation != seek_generation or state is None:
                    continue
                age = (time.perf_counter()-captured)*1000 + (current_header or {}).get('age_at_receive_ms', 0)
                if ros_mode and age > config['source']['timeout_s']*1000:
                    stabilizer.reset()
                    emit(decision.fail('inference_timeout'), frame, timestamp, event='error', age_ms=age)
                    continue
            except Exception as exc:
                stabilizer.reset()
                emit(decision.fail('inference_failed: ' + str(exc)), frame, timestamp, event='error')
                logging.exception('inference failure; stopping runtime')
                raise
            if ros_mode:
                with source.cv:
                    input_error = source.errors.popleft() if source.errors else None
                if input_error is not None:
                    stabilizer.reset()
                    emit(decision.fail('input_failed: ' + input_error), frame, timestamp, event='error')
                    continue
            processed += 1
            # Start the next inference while this frame is drawn and displayed.
            # The source already owns a single latest-frame slot.
            if not ros_mode and not paused and not stopped and (not config['monitor']['max_frames'] or processed < config['monitor']['max_frames']):
                next_kind, next_item = source.get(timeout=.001)
                if next_kind == 'frame':
                    next_frame, next_image, next_timestamp, next_captured, next_input_ms, next_header = next_item
                    if rotate:
                        next_image = cv2.rotate(next_image, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
                    next_item = (next_frame, next_image, next_timestamp, next_captured, next_input_ms, next_header)
                    prefetched = (next_kind, next_item, (executor.submit(infer_and_decide, next_image, seek_generation, next_input_ms, rotate), seek_generation))
                elif next_kind != 'waiting':
                    prefetched = (next_kind, next_item, None)
            if not emit(state, frame, timestamp, stages, image, result, age_ms=age):
                break
            apply_key(sinks.key)
            sinks.key = -1
            while paused and not stopped and (sinks.display or control):
                if service_control():
                    break
                if sinks.display:
                    if apply_key(cv2.waitKeyEx(50)):
                        break
                else:
                    time.sleep(.05)
            if config['monitor']['max_frames'] and processed >= config['monitor']['max_frames']:
                break
        if stopped or ros_mode:
            stabilizer.reset()
            emit(decision.fail('shutdown_signal' if stopped else 'runtime_finished'), event='shutdown')
        return processed
    finally:
        command_stop.set()
        if command_thread:
            command_thread.join()
        if source and source.thread.ident:
            source.close()
        if executor:
            executor.shutdown(wait=True)
        if backend:
            backend.close()
        sinks.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


# 경로·입력 종류·백엔드의 편리한 CLI 덮어쓰기를 지원한다.
def main():
    p = argparse.ArgumentParser(description='AGX vision runtime')
    p.add_argument('--config', default='config/runtime.yaml')
    p.add_argument('--source', choices=['v4l2','csi','rtsp','file','dir','ros2'])
    p.add_argument('--uri')
    p.add_argument('--backend', choices=['torch','onnx','tensorrt'])
    p.add_argument('--max-frames', type=int)
    p.add_argument('--rotate', type=int, choices=[0, 90, 180, 270])
    p.add_argument('--speed', type=float)
    p.add_argument('--window-width', type=int)
    p.add_argument('--control-stdio', action='store_true')
    argv = sys.argv[1:]
    if '--ros-args' in argv:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(args=sys.argv)[1:]
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    c = load_config(a.config)
    if a.source:
        c['source']['type'] = a.source
    if a.uri:
        c['source']['uri'] = a.uri
    if a.backend:
        c['backend']['type'] = a.backend
    if a.max_frames is not None:
        c['monitor']['max_frames'] = a.max_frames
    if a.rotate is not None:
        c['source']['rotate'] = a.rotate
    if a.speed is not None:
        c['source']['speed'] = a.speed
    if a.window_width is not None:
        if a.window_width <= 0:
            p.error('--window-width must be positive')
        c['sinks']['display']['width'] = a.window_width
    run(c, control=a.control_stdio)


if __name__ == '__main__':
    main()
