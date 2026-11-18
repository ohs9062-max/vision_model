"""Simple launcher for the existing file runtime."""
import os
import json
import queue
import threading
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
START_DIR = str(Path.home() / 'vision_videos') if (Path.home() / 'vision_videos').is_dir() else '/home/hs/다운로드/영상/과수원'

def selected_backend():
    import platform
    if platform.machine() != 'aarch64':
        return 'torch'
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    version = ROOT / 'weights' / config['active_model']
    return 'tensorrt' if all((version / f'{kind}.fp16.engine').is_file() for kind in ('seg', 'det')) else 'torch'


def make_command(path, rotate, speed, config_path, python=sys.executable):
    return [python, '-m', 'vision_model.runtime', '--config', str(config_path),
            '--source', 'file', '--uri', str(path), '--backend', selected_backend(),
            '--rotate', str(rotate), '--speed', str(speed), '--control-stdio']


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        return True
    return False


def readable_video(path):
    if not Path(path).is_file() or not os.access(path, os.R_OK):
        return False
    cap = cv2.VideoCapture(str(path))
    try:
        return cap.isOpened() and cap.read()[0]
    finally:
        cap.release()


class Player:
    def __init__(self, root):
        self.root, self.process, self.config_path = root, None, None
        self.path = ''
        self.selected = tk.StringVar(value='선택된 영상 없음')
        self.status = tk.StringVar(value='영상을 선택하세요.')
        self.rotate = tk.StringVar(value='0')
        self.speed = tk.StringVar(value='1.0')
        self.position = tk.StringVar(value='00:00 / 00:00')
        self.reports = queue.Queue()
        self.last_stderr = ''
        self.last_progress = time.monotonic()
        self.last_position = None
        self.termination_started = None
        self.restart_requested = None
        self.runtime_stopping_at = None
        self.stalled = False
        root.title('영상 추론 재생')
        frame = ttk.Frame(root, padding=12)
        frame.pack(fill='both', expand=True)
        ttk.Button(frame, text='영상 파일 열기', command=self.choose).pack(anchor='w')
        ttk.Label(frame, textvariable=self.selected, wraplength=440).pack(anchor='w', pady=6)
        buttons = ttk.Frame(frame)
        buttons.pack(anchor='w', pady=8)
        ttk.Button(buttons, text='재생', command=self.play).pack(side='left', padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text='정지', command=self.stop)
        self.stop_button.pack(side='left')
        controls = ttk.Frame(frame)
        controls.pack(anchor='w')
        self.playback_buttons = []
        for label, action in [('<', lambda: self.send('back5')), ('>', lambda: self.send('forward5')),
                              ('+', lambda: self.change_speed(1)), ('-', lambda: self.change_speed(-1)),
                              ('일시정지', lambda: self.send('pause')), ('전환', lambda: self.send('fullscreen')),
                              ('종료', self.stop)]:
            button = ttk.Button(controls, text=label, command=action)
            button.pack(side='left')
            self.playback_buttons.append(button)
        extra = ttk.Frame(frame)
        extra.pack(anchor='w', pady=5)
        for label, action in [('30초 뒤로', lambda: self.send('back30')),
                              ('30초 앞으로', lambda: self.send('forward30')),
                              ('화면 회전', self.turn)]:
            ttk.Button(extra, text=label, command=action).pack(side='left')
        ttk.Label(extra, textvariable=self.rotate).pack(side='left', padx=5)
        ttk.Label(extra, textvariable=self.speed).pack(side='left', padx=5)
        ttk.Label(frame, text='방향키: 이동  스페이스: 일시정지  +/−: 배속  f: 전환  r: 회전  q/Esc: 종료').pack(anchor='w')
        actions = {'Left': lambda: self.send('back5'), 'Right': lambda: self.send('forward5'),
                   'Up': lambda: self.send('forward30'), 'Down': lambda: self.send('back30'),
                   'space': lambda: self.send('pause'), 'plus': lambda: self.change_speed(1),
                   'equal': lambda: self.change_speed(1), 'minus': lambda: self.change_speed(-1),
                   'bracketright': lambda: self.change_speed(1), 'bracketleft': lambda: self.change_speed(-1),
                   'r': self.turn, 'f': lambda: self.send('fullscreen'),
                   '1': lambda: self.send('reset_window'), 'q': self.stop, 'Escape': self.stop}
        for key, action in actions.items():
            root.bind('<KeyPress-' + key + '>', lambda event, action=action: (action(), 'break')[1])
        ttk.Label(frame, textvariable=self.position).pack(anchor='w', pady=(8, 0))
        ttk.Label(frame, textvariable=self.status, wraplength=440).pack(anchor='w')
        self.update_buttons()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(300, self.poll)

    # 재생 상태에 맞춰 이동·일시정지·정지 버튼을 활성화한다.
    def update_buttons(self):
        active = self.process is not None and self.process.poll() is None
        for button in self.playback_buttons + [self.stop_button]:
            button.configure(state='normal' if active else 'disabled')

    # 실행 중인 프로세스에 한 줄 명령을 보낸다.
    def send(self, command):
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    # 재생 전에는 시작 각도를 선택하고 재생 중에는 회전 명령을 보낸다.
    def turn(self):
        if self.process is not None and self.process.poll() is None:
            self.send('rotate')
        else:
            self.rotate.set(str((int(self.rotate.get()) + 90) % 360))

    # 재생 전에는 시작 배속을 선택하고 재생 중에는 단계 명령을 보낸다.
    def change_speed(self, direction):
        if self.process is not None and self.process.poll() is None:
            self.send('faster' if direction > 0 else 'slower')
        else:
            steps = (.25, .5, 1., 2., 4.)
            index = steps.index(float(self.speed.get()))
            self.speed.set(str(steps[max(0, min(len(steps)-1, index + direction))]))

    # 런타임의 상태 줄을 읽어 Tk 스레드에 전달한다.
    def read_reports(self, process):
        for line in process.stdout:
            try:
                report = json.loads(line)
                if report.get('event') == 'stopping':
                    if process is self.process:
                        self.runtime_stopping_at = report['time']
                else:
                    self.reports.put((process, report))
            except (ValueError, TypeError):
                continue

    def read_stderr(self, process):
        for line in process.stderr:
            line = line.strip()
            if line and process is self.process:
                self.last_stderr = line

    # 영상 선택 대화상자에서 재생할 파일을 고른다.
    def choose(self):
        path = filedialog.askopenfilename(initialdir=START_DIR, title='영상 파일 선택',
            filetypes=[('영상 파일', '*.mp4 *.mov *.MOV *.avi *.mkv'), ('모든 파일', '*')])
        if path:
            self.path = path
            self.selected.set(Path(path).name)
            self.status.set('재생 버튼을 누르세요.')

    def play(self):
        if self.process is not None:
            self.restart_requested = time.monotonic()
            self.status.set('재생 중입니다. 먼저 정지하세요.')
            return
        try:
            if not self.path or not readable_video(self.path):
                self.status.set('영상 파일을 열 수 없습니다.')
                return
            config = yaml.safe_load((ROOT / 'config/runtime.yaml').read_text())
            config['sinks']['display']['enabled'] = True
            gpu = torch.cuda.is_available()
            if not gpu:
                config['backend']['device'] = 'cpu'
            output = ROOT / 'outputs/runtime'
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.yaml', prefix='player_', dir=output, delete=False) as f:
                self.config_path = Path(f.name)
                yaml.safe_dump(config, f, allow_unicode=True)
            self.process = subprocess.Popen(make_command(self.path, self.rotate.get(), self.speed.get(), self.config_path), cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self.last_stderr = ''
            self.last_progress = time.monotonic()
            self.last_position = None
            self.termination_started = None
            self.stalled = False
            self.restart_requested = None
            self.runtime_stopping_at = None
            self.stdout_thread = threading.Thread(target=self.read_reports, args=(self.process,), daemon=True)
            self.stdout_thread.start()
            self.stderr_thread = threading.Thread(target=self.read_stderr, args=(self.process,), daemon=True)
            self.stderr_thread.start()
            self.update_buttons()
            self.status.set('재생 중' if gpu else 'GPU 없음: CPU로 재생 중')
        except Exception as exc:
            self.cleanup()
            self.status.set('실행 오류: ' + str(exc).replace('\n', ' '))

    def cleanup(self):
        if self.config_path is not None:
            self.config_path.unlink(missing_ok=True)
            self.config_path = None

    def stop(self):
        if stop_process(self.process):
            self.restart_requested = None
            self.status.set('정지 중...')
            self.stopping = True
            self.termination_started = time.monotonic()

    # 상태 보고를 0.5초 간격으로 표시하고 종료를 정리한다.
    def poll(self):
        while True:
            try:
                process, report = self.reports.get_nowait()
            except queue.Empty:
                break
            if process is self.process:
                def clock(seconds):
                    seconds = max(0, int(seconds))
                    return f'{seconds // 60:02d}:{seconds % 60:02d}'
                if report['paused'] or report['position'] != self.last_position:
                    self.last_progress = time.monotonic()
                    self.last_position = report['position']
                self.position.set(clock(report['position']) + ' / ' + clock(report['duration']))
                self.rotate.set(str(report['rotate']))
                self.speed.set(str(report['speed']))
                self.status.set('일시정지' if report['paused'] else '재생 중')
        if self.process is not None:
            rc = self.process.poll()
            now = time.monotonic()
            if rc is None and self.termination_started is None and now - self.last_progress > 45:
                self.stalled = True
                self.process.terminate()
                self.termination_started = now
                self.status.set('재생 응답 없음: 정리 중...')
            if rc is None and self.termination_started is not None and now - self.termination_started > 5:
                self.process.kill()
            if rc is not None:
                self.stdout_thread.join(timeout=1)
                self.stderr_thread.join(timeout=1)
                stopped = getattr(self, 'stopping', False)
                request = self.restart_requested
                notice = self.runtime_stopping_at
                restart = (request is not None and not self.stalled and
                           (stopped or (rc == 0 and notice is not None and
                                        notice - .25 <= request)))
                self.stopping = False
                self.process = None
                self.restart_requested = None
                self.runtime_stopping_at = None
                self.cleanup()
                self.update_buttons()
                message = ('재생 응답 없음' if self.stalled else
                           '정지됨' if stopped else '재생 종료' if rc == 0 else f'재생 프로세스 오류 (종료 코드 {rc})')
                self.status.set(message + (': ' + self.last_stderr if self.last_stderr else ''))
                if restart:
                    self.play()
        self.root.after(500, self.poll)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.cleanup()
        self.root.destroy()


def main():
    root = tk.Tk()
    Player(root)
    root.mainloop()


if __name__ == '__main__':
    main()
