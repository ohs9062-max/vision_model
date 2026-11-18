#!/usr/bin/env python3
"""ROS2 runtime regression checks without models, devices or DDS sockets."""
import copy
from collections import deque
from pathlib import Path
import sys
import threading
import tempfile
import cv2
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_model.runtime.decision import Decision
from vision_model.runtime.sources import Source
from vision_model.runtime.stabilizer import Stabilizer
from vision_model.runtime.ros2_video import format_decision
from vision_model.runtime.__main__ import run


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/runtime.yaml').read_text())

    def setUp(self):
        self.c = copy.deepcopy(self.base)
        self.c['source'].update(type='ros2', timeout_s=.05)
        self.c['sinks']['ros2']['enabled'] = True
        self.result = dict(mask=np.zeros((10, 10), np.uint8), boxes=[])

    def test_existing_directory_runtime_with_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            for i in range(2):
                cv2.imwrite(str(Path(directory) / f'{i}.png'), np.zeros((10, 10, 3), np.uint8))
            self.c['source'].update(type='dir', uri=directory, latest_only=False)
            self.c['sinks']['ros2']['enabled'] = False
            self.c['monitor']['max_frames'] = 2
            backend = Mock(side_effect=lambda image: (dict(self.result), dict(inference=0., postprocess=0.)))
            sink = Mock(display=False, key=-1)
            sink.emit.return_value = True
            with patch('vision_model.runtime.__main__.Backend', return_value=backend), patch('vision_model.runtime.__main__.Sinks', return_value=sink):
                self.assertEqual(run(self.c), 2)
            self.assertEqual(backend.call_count, 2)
            self.assertEqual(sink.emit.call_count, 2)

    def test_stabilizer_before_after_uses_same_spatial_policy(self):
        d = Decision(dict(self.c['decision'], require_validated=True))
        stabilizer = Stabilizer(self.c['stabilizer'])
        hazard = dict(self.result, boxes=[([2, 6, 8, 10], .9, 3)])
        before = dict(d.__dict__)
        self.assertEqual(d.evaluate(hazard)['candidate'], 'STOP')
        self.assertEqual(d.__dict__, before)
        tracked = dict(hazard, boxes=stabilizer.update(hazard['boxes']))
        state = d(tracked)
        self.assertEqual(state['candidate'], 'GO')
        self.assertEqual(state['decision'], 'STOP')
        tracked['boxes'] = stabilizer.update(hazard['boxes'])
        self.assertEqual(d(tracked)['candidate'], 'STOP')

    def test_terminal_keeps_candidate_and_published_separate(self):
        text = format_decision(dict(frame=1, candidate='GO', decision='STOP', decision_valid=False,
                                    inference_status='ok', event='frame', boxes=[],
                                    pre_stabilizer=dict(candidate='SLOW', box_count=1)))
        self.assertIn('Candidate: GO', text)
        self.assertIn('Published: STOP', text)
        self.assertIn('Stabilizer: SLOW -> GO', text)
        self.assertIn('Valid: false', text)

    def test_policy_validation_and_legacy_rules(self):
        d = Decision(dict(self.c['decision'], require_validated=True))
        for _ in range(20):
            state = d(self.result)
            self.assertEqual(state['decision'], 'STOP')
            self.assertEqual(state['candidate'], 'GO')
            self.assertEqual(state['reasons'], ['policy_unvalidated'])
        d = Decision(dict(self.c['decision'], policy_validated=True, require_validated=True))
        for _ in range(self.c['decision']['go_frames']):
            state = d(self.result)
        self.assertEqual(state['decision'], 'GO')
        hazard = dict(self.result, boxes=[([2, 6, 8, 10], .9, 3)])
        for _ in range(self.c['decision']['stop_frames']):
            state = d(hazard)
        self.assertEqual(state['decision'], 'STOP')
        self.assertIn('obstacle_near', state['reasons'])
        self.assertEqual(d.fail('model_error')['decision'], 'STOP')

    def source_and_message(self):
        source = Source(self.c['source'])
        source.node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_010_000_000)))
        msg = SimpleNamespace(encoding='rgb8', height=1, width=2, step=8,
                              data=bytes([1, 2, 3, 4, 5, 6, 0, 0]),
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0), frame_id='optical'))
        return source, msg

    def test_conversion_header_stride_and_latest_frame(self):
        source, msg = self.source_and_message()
        source._ros_callback(msg)
        source._ros_callback(msg)
        kind, item = source.get(timeout=.001)
        self.assertEqual(kind, 'frame')
        self.assertEqual(source.dropped, 1)
        np.testing.assert_array_equal(item[1], [[[3, 2, 1], [6, 5, 4]]])
        self.assertEqual(item[2], 10.)
        self.assertEqual(item[5]['frame_id'], 'optical')
        self.assertEqual(item[5]['stamp'], dict(sec=10, nanosec=0))

    def test_invalid_and_stale_images_clear_pending_frame(self):
        for mode in ('stride', 'encoding', 'stale', 'zero', 'future'):
            with self.subTest(mode=mode):
                source, msg = self.source_and_message()
                source._ros_callback(msg)
                if mode == 'stride':
                    msg.step = 1
                elif mode == 'encoding':
                    msg.encoding = '32FC1'
                else:
                    msg.header.stamp.sec = {'stale': 9, 'zero': 0, 'future': 11}[mode]
                source._ros_callback(msg)
                self.assertEqual(source.get(timeout=.001)[0], 'error')
                self.assertIsNone(source.slot)

    def test_polling_timeout_is_not_hidden_by_short_waits(self):
        source = Source(self.c['source'])
        source.last_received = time.perf_counter() - 1
        self.assertEqual(source.get(timeout=.001), ('error', 'input_timeout'))

    def exercise_run(self, backend_kind):
        packets = []
        c = self.c
        if backend_kind != 'ok':
            c['decision'].update(policy_validated=True, go_frames=1)
        c['monitor']['max_frames'] = 1
        header = dict(stamp=dict(sec=10, nanosec=20), frame_id='optical', age_at_receive_ms=0)
        result = self.result

        class Input:
            def __init__(self, config):
                self.cv = threading.Condition()
                self.stop = threading.Event()
                self.errors = deque()
                self.dropped = 0
                self.thread = threading.Thread(target=lambda: None)
                self.calls = 0
                Input.instance = self

            def get(self, timeout=None):
                self.calls += 1
                if self.calls > 1:
                    return 'eof', None
                return 'frame', (1, np.zeros((10, 10, 3), np.uint8), 10.00000002, time.perf_counter(), 0., header)

            def controls(self, speed, paused):
                pass

            def close(self):
                self.thread.join()

        class Model:
            loads = 0

            def __init__(self, config):
                Model.loads += 1

            def __call__(self, image):
                if backend_kind == 'error':
                    raise RuntimeError('injected model failure')
                if backend_kind == 'delay':
                    time.sleep(.08)
                if backend_kind == 'input_error':
                    with Input.instance.cv:
                        Input.instance.errors.append('conversion_failure')
                    time.sleep(.03)
                if backend_kind == 'invalid_mask':
                    return dict(mask=np.full((10, 10), 255, np.uint8), boxes=[]), dict(inference=0., postprocess=0.)
                return dict(result), dict(inference=0., postprocess=0.)

            def close(self):
                pass

        class Output:
            display = False
            key = -1

            def __init__(self, config):
                pass

            def emit(self, packet, image=None, result=None):
                packets.append(packet)
                return True

            def handle_window_key(self, key):
                pass

            def close(self):
                pass

        with patch('vision_model.runtime.__main__.Source', Input), patch('vision_model.runtime.__main__.Backend', Model), patch('vision_model.runtime.__main__.Sinks', Output):
            if backend_kind in ('error', 'invalid_mask'):
                with self.assertRaisesRegex((RuntimeError, ValueError), 'injected|invalid segmentation'):
                    run(c)
            else:
                run(c)
        self.assertEqual(Model.loads, 1)
        self.assertTrue(packets)
        self.assertFalse(any(p['decision'] == 'GO' for p in packets))
        return packets

    def test_runtime_metadata_and_unvalidated_policy(self):
        packets = self.exercise_run('ok')
        packet = next(p for p in packets if p['event'] == 'frame')
        self.assertEqual(packet['header']['stamp']['nanosec'], 20)
        self.assertEqual(packet['last_success_timestamp'], 10.00000002)
        self.assertEqual(packet['inference_status'], 'ok')
        self.assertFalse(packet['decision_valid'])
        self.assertIn('policy_unvalidated', packet['reasons'])
        self.assertIn('pre_stabilizer', packet)

    def test_inference_error_never_emits_go(self):
        packets = self.exercise_run('error')
        self.assertTrue(any('inference_failed' in p['reasons'][0] for p in packets))

    def test_invalid_model_mask_never_emits_go(self):
        packets = self.exercise_run('invalid_mask')
        self.assertTrue(any('invalid segmentation mask' in p['reasons'][0] for p in packets))

    def test_input_error_during_inference_discards_result(self):
        packets = self.exercise_run('input_error')
        self.assertTrue(any('conversion_failure' in p['reasons'][0] for p in packets))
        self.assertFalse(any(p['event'] == 'frame' for p in packets))

    def test_late_inference_is_discarded(self):
        packets = self.exercise_run('delay')
        self.assertTrue(any(p['reasons'] == ['inference_timeout'] for p in packets))
        self.assertFalse(any(p['event'] == 'frame' for p in packets))


if __name__ == '__main__':
    unittest.main(verbosity=2)
