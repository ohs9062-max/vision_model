# JSON·UDP·ROS2·영상·화면 출력을 동시에 제공한다.
import json
import logging
import os
from pathlib import Path
import socket
import cv2
from ..draw import draw
from .config import make_qos


class Sinks:
    # 선택된 출력만 열고 ROS 의존성은 필요할 때 불러온다.
    def __init__(self, config):
        self.c = config
        self.file, self.sock, self.video, self.node, self.context = None, None, None, None, None
        self.key = -1
        self.window_open = False
        self.fullscreen = False
        self.window_shape = None
        self.display = config['display']['enabled'] and bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        if config['display']['enabled'] and not self.display:
            logging.warning('display disabled: no display server')
        try:
            if config['json']['enabled']:
                path = Path(config['json']['path'])
                path.parent.mkdir(parents=True, exist_ok=True)
                self.file = path.open('w', buffering=1)
            if config['udp']['enabled']:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if config['ros2']['enabled']:
                import rclpy
                from rclpy.context import Context
                from std_msgs.msg import String
                from sensor_msgs.msg import Image
                self.context = Context()
                rclpy.init(context=self.context)
                self.node = rclpy.create_node('vision_output', context=self.context)
                c = config['ros2']
                self.pub_decision = self.node.create_publisher(String, c['decision_topic'], make_qos(c['qos']['decision']))
                self.pub_boxes = self.node.create_publisher(String, c['boxes_topic'], make_qos(c['qos']['boxes']))
                self.pub_mask = self.node.create_publisher(Image, c['mask_topic'], make_qos(c['qos']['mask']))
        except Exception:
            self.close()
            raise

    # 표시 창을 만들고 현재 영상 비율로 시작 크기를 적용한다.
    def resize_window(self, shape):
        width = self.c['display']['width']
        height = max(1, round(shape[0] * width / shape[1]))
        if not self.window_open:
            cv2.namedWindow('vision runtime', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            self.window_open = True
        cv2.resizeWindow('vision runtime', width, height)
        self.window_shape = shape[:2]

    # 전체화면 전환과 시작 크기 복원 키를 처리한다.
    def handle_window_key(self, key):
        if not self.display or not self.window_open:
            return
        if key == ord('f'):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty('vision runtime', cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord('1'):
            if self.fullscreen:
                self.fullscreen = False
                cv2.setWindowProperty('vision runtime', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            self.resize_window(self.window_shape)

    # 프레임 메타데이터와 선택적 마스크·주석 영상을 전송한다.
    def emit(self, packet, image=None, result=None):
        payload = json.dumps(packet, ensure_ascii=False, allow_nan=False)
        if self.file:
            self.file.write(payload + '\n')
        if self.sock:
            data = payload.encode()
            if len(data) > 65507:
                raise ValueError('UDP JSON exceeds datagram limit; reduce box count or use JSON/ROS2')
            self.sock.sendto(data, (self.c['udp']['host'], self.c['udp']['port']))
        if self.node:
            from std_msgs.msg import String
            from sensor_msgs.msg import Image
            self.pub_decision.publish(String(data=payload))
            self.pub_boxes.publish(String(data=json.dumps(dict(timestamp=packet['timestamp'], frame=packet['frame'], boxes=packet['boxes'], header=packet.get('header'), inference_status=packet.get('inference_status'), detected_classes=packet.get('detected_classes')))))
            if result is not None:
                mask = result['mask']
                msg = Image()
                header = packet.get('header')
                if header:
                    msg.header.stamp.sec = header['stamp']['sec']
                    msg.header.stamp.nanosec = header['stamp']['nanosec']
                    msg.header.frame_id = header['frame_id']
                else:
                    ns = round(packet['timestamp'] * 1e9)
                    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
                msg.height, msg.width = mask.shape
                msg.encoding, msg.step = 'mono8', mask.shape[1]
                msg.data = mask.tobytes()
                self.pub_mask.publish(msg)
        if image is not None and result is not None and (self.c['video']['enabled'] or self.display):
            panel = draw(image, result)
            cv2.putText(panel, packet['decision'] + ' ' + ','.join(packet['reasons']), (15, panel.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
            cv2.putText(panel, f"{packet['speed']:g}x  {packet['rotate']} deg  {packet['fps']:.1f} FPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2)
            if self.c['video']['enabled']:
                # 저장 영상만 설정 폭으로 축소하고 화면에는 원본 해상도를 유지한다.
                width = self.c['video']['width']
                size = (width, max(2, round(panel.shape[0]*width/panel.shape[1])//2*2))
                video_panel = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
                if self.video is None:
                    path = Path(self.c['video']['path'])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.c['video']['fps'], (video_panel.shape[1],video_panel.shape[0]))
                    if not self.video.isOpened():
                        raise OSError('cannot open mp4 sink')
                self.video.write(video_panel)
            if self.display:
                panel = panel.copy()
                try:
                    if not self.window_open or (self.window_shape[0] * panel.shape[1] != panel.shape[0] * self.window_shape[1]):
                        self.resize_window(panel.shape)
                    cv2.imshow('vision runtime', panel)
                    # Event polling is handled by the runtime main thread.
                except cv2.error as exc:
                    logging.warning('display disabled: %s', exc)
                    self.display = False
        return True

    # 출력 파일·소켓·ROS 노드를 정리한다.
    def close(self):
        if self.file:
            self.file.close()
        if self.sock:
            self.sock.close()
        if self.video:
            self.video.release()
        if self.display:
            cv2.destroyAllWindows()
        if self.node:
            self.node.destroy_node()
        if self.context:
            self.context.try_shutdown()
