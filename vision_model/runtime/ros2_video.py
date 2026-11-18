"""Replay original file frames through Source; inspect existing ROS decision JSON.

Neither command creates an inference backend. All models remain in runtime.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import sys
import threading
import time

from .__main__ import load_config
from .sources import Source


def write_record(stream, record):
    if stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def publish(config, frame_id, max_frames=0, wait_s=0., output=None):
    import rclpy
    from rclpy.context import Context
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    if config['source']['type'] != 'file':
        raise ValueError('publisher requires file source')
    if not Path(config['source']['uri']).is_file():
        raise FileNotFoundError(config['source']['uri'])
    context = Context()
    node = source = stream = None
    stopped = threading.Event()
    old_handlers = {}
    first_tick, last_tick, count, last_frame = None, None, 0, 0
    reason = 'interrupted'
    try:
        rclpy.init(context=context)
        node = rclpy.create_node('vision_video_camera', context=context)
        publisher = node.create_publisher(Image, config['source']['topic'],
                                         QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: stopped.set())
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('w')
        deadline = time.monotonic() + wait_s
        while wait_s and not publisher.get_subscription_count() and not stopped.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError('no camera subscriber within wait timeout')
            stopped.wait(.05)
        if stopped.is_set():
            return None
        source = Source(config['source'])
        source.thread.start()
        while not stopped.is_set():
            kind, item = source.get(timeout=.05)
            if kind == 'waiting':
                continue
            if kind == 'eof':
                reason = 'eof'
                break
            if kind == 'error':
                raise RuntimeError('video input failed: ' + str(item))
            frame, image, _, captured, _, _ = item
            # Source supplies the decoded original. No drawing/model/resize here.
            msg = Image()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            msg.height, msg.width = image.shape[:2]
            msg.encoding, msg.step = 'bgr8', image.shape[1] * 3
            msg.data = image.tobytes()
            publisher.publish(msg)
            now = time.perf_counter()
            if first_tick is None:
                first_tick = now
            last_tick = now
            count += 1
            last_frame = frame
            write_record(stream, dict(event='published_frame', frame=frame,
                                     source_time_s=(frame-1)/source.fps,
                                     stamp=dict(sec=msg.header.stamp.sec, nanosec=msg.header.stamp.nanosec),
                                     frame_id=frame_id, width=msg.width, height=msg.height,
                                     publish_elapsed_s=now-first_tick, frame_age_ms=(now-captured)*1000))
            if max_frames and count >= max_frames:
                reason = 'max_frames'
                break
        summary = dict(event='publisher_summary', source=config['source']['uri'],
                       topic=config['source']['topic'], nominal_fps=source.fps,
                       speed=source.speed, published_frames=count, last_source_frame=last_frame,
                       source_dropped_frames=source.dropped, end_reason=reason,
                       publish_fps=(count-1)/(last_tick-first_tick) if count > 1 else 0.)
        write_record(stream, summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    finally:
        if source and source.thread.ident:
            source.close()
        if stream:
            stream.close()
        if node:
            node.destroy_node()
        if context.ok():
            context.try_shutdown()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def format_decision(packet):
    """Display existing candidate/decision keys, plus the measured tracking effect."""
    before = packet.get('pre_stabilizer', {})
    return (f"Frame {packet.get('frame', 0):06d} | stamp={packet.get('timestamp')} "
            f"Candidate: {packet.get('candidate')} | Published: {packet.get('decision')} "
            f"| Valid: {str(packet.get('decision_valid')).lower()} "
            f"| inference={packet.get('inference_status')} "
            f"| event={packet.get('event')}\n"
            f"  Stabilizer: {before.get('candidate', '-')} -> {packet.get('candidate')} "
            f"| boxes: {before.get('box_count', '-')} -> {len(packet.get('boxes', []))} "
            f"| classes={packet.get('detected_classes', [])}\n"
            f"  Candidate reasons={packet.get('candidate_reasons', packet.get('reasons', []))} "
            f"| Published reasons={packet.get('reasons', [])} "
            f"| ROI={packet.get('roi_ratios', {})} "
            f"| fps={packet.get('fps', 0):.2f} "
            f"| age_ms={packet.get('frame_age_ms', 0):.1f} "
            f"| dropped={packet.get('dropped_frames', 0)}")


def watch(config, max_frames=0, timeout_s=0., output=None):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String

    context = Context()
    node = executor = stream = None
    stopped = threading.Event()
    old_handlers = {}
    candidates, published = Counter(), Counter()
    count = errors = dropped = 0
    try:
        rclpy.init(context=context)
        node = rclpy.create_node('vision_decision_watch', context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('w')
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, lambda *_: stopped.set())

        def receive(msg):
            nonlocal count, errors, dropped
            packet = json.loads(msg.data)
            write_record(stream, packet)
            print(format_decision(packet), flush=True)
            if packet.get('event') == 'frame':
                count += 1
                candidates[packet['candidate']] += 1
                published[packet['decision']] += 1
            else:
                errors += packet.get('event') == 'error'
            dropped = max(dropped, packet.get('dropped_frames', 0))
            if max_frames and count >= max_frames:
                stopped.set()

        node.create_subscription(String, config['sinks']['ros2']['decision_topic'], receive, 10)
        started = time.monotonic()
        while not stopped.is_set() and (not timeout_s or time.monotonic()-started < timeout_s):
            executor.spin_once(timeout_sec=.1)
        summary = dict(event='watch_summary', processed_frames=count, candidates=dict(candidates),
                       published=dict(published), errors=errors, source_dropped_frames=dropped)
        write_record(stream, summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    finally:
        if stream:
            stream.close()
        if executor:
            executor.shutdown()
        if node:
            node.destroy_node()
        if context.ok():
            context.try_shutdown()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    replay = sub.add_parser('publish', help='publish original video frames without inference')
    replay.add_argument('--uri', help='override source.uri with an MP4 path')
    replay.add_argument('--frame-id', required=True, help='camera coordinate frame agreed with the robot team')
    replay.add_argument('--speed', type=float, choices=(.25, .5, 1., 2., 4.))
    replay.add_argument('--wait-for-subscriber', type=float, default=0.)
    monitor = sub.add_parser('watch', help='display candidate and published decision JSON')
    monitor.add_argument('--timeout', type=float, default=0.)
    for command in (replay, monitor):
        command.add_argument('--config', default='config/runtime.yaml')
        command.add_argument('--max-frames', type=int, default=0)
        command.add_argument('--output', help='optional JSONL evidence path')
    argv = sys.argv[1:]
    if '--ros-args' in argv:
        from rclpy.utilities import remove_ros_args
        argv = remove_ros_args(args=sys.argv)[1:]
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    config = load_config(args.config)
    if args.command == 'publish':
        if args.wait_for_subscriber < 0:
            parser.error('--wait-for-subscriber must be nonnegative')
        config['source']['type'] = 'file'
        if args.uri:
            config['source']['uri'] = args.uri
        if args.speed:
            config['source']['speed'] = args.speed
        publish(config, args.frame_id, args.max_frames, args.wait_for_subscriber, args.output)
    else:
        if args.timeout < 0:
            parser.error('--timeout must be nonnegative')
        watch(config, args.max_frames, args.timeout, args.output)


if __name__ == '__main__':
    main()
