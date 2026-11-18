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
from vision_model.runtime.geometry import GroundGeometry
from vision_model.runtime.sources import Source
from vision_model.runtime.sinks import Sinks
from vision_model.runtime.stabilizer import Stabilizer
from vision_model.runtime.stabilizer import create_processors
from vision_model.runtime.ros2_video import format_decision
from vision_model.runtime.__main__ import run, load_config
from vision_model.runtime.config import validate_runtime, make_qos

CONFIG = load_config(Path(__file__).resolve().parents[1] / 'config/runtime.yaml')


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = CONFIG

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
        d = Decision(dict(self.c['decision'], geometry=self.c['geometry'], require_validated=True))
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
        d = Decision(dict(self.c['decision'], geometry=self.c['geometry'], require_validated=True))
        for _ in range(20):
            state = d(self.result)
            self.assertEqual(state['decision'], 'STOP')
            self.assertEqual(state['candidate'], 'GO')
            self.assertEqual(state['reasons'], ['policy_unvalidated'])
        d = Decision(dict(self.c['decision'], geometry=self.c['geometry'], policy_validated=True, require_validated=True))
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
        self.assertEqual(packet['hazards_frame_id'], 'base_link')
        self.assertFalse(packet['geometry']['calibrated'])
        self.assertEqual(packet['header']['frame_id'], 'optical')
        self.assertEqual(packet['hazards'], [])
        for p in packets:
            self.assertIn('geometry', p)
            self.assertIn('hazards', p)

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


class GeometryChecks(unittest.TestCase):
    def geometry(self, **kwargs):
        c = copy.deepcopy(CONFIG['geometry'])
        c.update(enabled=True, pitch_deg=12.)
        c.update(kwargs)
        return GroundGeometry(c)

    def pixel(self, g, x, y, w=640, h=480):
        # Independent ground -> optical camera transform (REP-103 y left).
        pitch = np.deg2rad(g.c['pitch_deg'])
        depth = x * np.cos(pitch) + g.c['height_m'] * np.sin(pitch)
        down = g.c['height_m'] * np.cos(pitch) - x * np.sin(pitch)
        fx, fy, cx, cy = g.intrinsics(w, h)
        return cx - fx * y / depth, cy + fy * down / depth

    def test_projection_roundtrip_and_rotations(self):
        g = self.geometry()
        for x, y in ((.5, .1), (1., -.2), (3., 0.), (4.5, .3)):
            u, v = self.pixel(g, x, y)
            for rotate, a, b, w, h in ((0, u, v, 640, 480), (90, 479-v, u, 480, 640),
                                        (180, 639-u, 479-v, 640, 480), (270, v, 639-u, 480, 640)):
                actual_x, actual_y, reason = g.project(a, b, w, h, rotate)
                self.assertIsNone(reason)
                np.testing.assert_allclose([actual_x, actual_y], [x, y], atol=1e-10)

    def test_null_horizon_range_and_invalid_pixel(self):
        g = self.geometry()
        fx, fy, cx, cy = g.intrinsics(640, 480)
        horizon = cy - fy * np.tan(np.deg2rad(12))
        self.assertEqual(g.project(cx, horizon-1, 640, 480), (None, None, 'above_horizon'))
        self.assertEqual(g.project(cx, horizon, 640, 480), (None, None, 'above_horizon'))
        u, v = self.pixel(g, 6., 0.)
        self.assertEqual(g.project(u, v, 640, 480), (None, None, 'beyond_max_range'))
        self.assertEqual(g.project(-1, 0, 640, 480)[2], 'pixel_outside_image')
        self.assertEqual(g.project(float('nan'), 0, 640, 480)[2], 'invalid_pixel')

    def test_stop_slow_none_and_overlap(self):
        g = self.geometry(robot_width_m=.6, margin_m=.15, stop_length_m=1., slow_length_m=2.5)
        for x, left, right, expected in ((.8,.2,-.2,'stop'), (1.,.2,-.2,'stop'),
                                         (1.5,.2,-.2,'slow'), (2.5,.2,-.2,'slow'),
                                         (3.,.2,-.2,'none'), (.8,.8,.5,'none'),
                                         (.8,.8,.4,'stop'), (.8,-.4,-.8,'stop'),
                                         (None,None,None,'none')):
            self.assertEqual(g.zone(x, left, right), expected)

    def test_table_and_configuration_contract(self):
        g = self.geometry(method='table')
        self.assertEqual(g.project(320, 400, 640, 480), (None, None, 'table_not_calibrated'))
        for c in (dict(method='unknown'), dict(height_m=0), dict(pitch_deg=float('nan')),
                  dict(fx=100), dict(table=[[.5, 1], [.4, 2]]), dict(table=[[-.1, 2]]),
                  dict(stop_length_m=3, slow_length_m=2)):
            with self.assertRaises(ValueError):
                self.geometry(**c)
        self.geometry(method='table', table=[[.5, 3], [.8, 1]])
        self.assertFalse(self.geometry().metadata()['calibrated'])

    def test_intrinsics_scale(self):
        g = self.geometry(fx=450, fy=460, cx=319.5, cy=239.5,
                          reference_width=640, reference_height=480)
        u, v = self.pixel(g, 1., .1)
        np.testing.assert_allclose(g.project((u+.5)*2-.5, (v+.5)*2-.5, 1280, 960)[:2], [1.,.1], atol=1e-10)

    def test_hazard_geometry_and_opt_in_only(self):
        c = copy.deepcopy(CONFIG)
        c['geometry'].update(enabled=True, pitch_deg=12.)
        _, d = create_processors(c)
        # Entire box outside ROI but inside the metric path.
        u, v = self.pixel(d.geometry, 2., .4)
        box = [u-2, v-5, u+2, v]
        r = dict(mask=np.zeros((480,640), np.uint8), boxes=[(box,.9,3)])
        state = d.evaluate(r)
        self.assertEqual(state['candidate'], 'GO')
        hazard = state['hazards'][0]
        self.assertEqual(hazard['class'], 'obstacle')
        self.assertEqual(hazard['frame_id'], 'base_link')
        self.assertTrue(hazard['range_valid'])
        np.testing.assert_allclose([hazard['x_m'],hazard['y_m']], [2.,.4], atol=1e-10)
        self.assertGreater(hazard['left_y_m'], hazard['right_y_m'])
        self.assertEqual(hazard['zone'], 'slow')
        c['geometry']['affect_candidate'] = True
        _, d = create_processors(c)
        self.assertEqual(d.evaluate(r)['candidate'], 'SLOW')
        c['decision']['require_validated'] = True
        _, d = create_processors(c)
        self.assertEqual(d(r)['decision'], 'STOP')
        self.assertEqual(d(r)['reasons'], ['policy_unvalidated'])

    def test_default_decision_sequence_unchanged(self):
        c = copy.deepcopy(CONFIG)
        _, new = create_processors(c)
        legacy = Decision(dict(c['decision'], geometry=c['geometry']))
        rng = np.random.default_rng(123)
        for _ in range(100):
            mask = rng.choice([0,1,2], size=(48,64), p=[.65,.2,.15]).astype(np.uint8)
            boxes = [([10,20,40,47], .9, int(rng.integers(4)))] if rng.random() > .3 else []
            r = dict(mask=mask, boxes=boxes)
            a, b = new(r), legacy(r)
            for k in ('decision', 'candidate', 'reasons', 'candidate_reasons', 'roi_ratios'):
                self.assertEqual(a[k], b[k])

    def test_hazard_null_reasons_and_stop(self):
        g = self.geometry()
        u, v = self.pixel(g, .8, 0.)
        h = g.hazards([([u-10,v-20,u+10,v],.8,0)], ['step'], 640,480)[0]
        self.assertEqual(h['zone'], 'stop')
        self.assertTrue(h['range_valid'])
        h = g.hazards([([300,0,340,50],.8,0)], ['step'], 640,480)[0]
        self.assertIsNone(h['x_m'])
        self.assertIsNone(h['y_m'])
        self.assertFalse(h['range_valid'])
        self.assertEqual(h['reason'], 'above_horizon')
        self.assertEqual(h['zone'], 'none')
        h = self.geometry(enabled=False).hazards([([300,350,340,400],.8,0)], ['step'],640,480)[0]
        self.assertEqual(h['reason'], 'geometry_disabled')

    def test_metric_policy_cannot_relax_roi_or_unlock(self):
        c = copy.deepcopy(CONFIG)
        c['geometry']['affect_candidate'] = True
        c['decision']['require_validated'] = True
        _, d = create_processors(c)
        r = dict(mask=np.full((48,64),2,np.uint8), boxes=[])
        for _ in range(20):
            state = d(r)
            self.assertEqual(state['candidate'], 'STOP')
            self.assertEqual(state['decision'], 'STOP')
            self.assertEqual(state['reasons'], ['policy_unvalidated'])


class ConfigurationChecks(unittest.TestCase):
    def test_every_yaml_key_is_required_before_io(self):
        def paths(value, prefix=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield prefix + (key,)
                    yield from paths(item, prefix + (key,))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    yield from paths(item, prefix + (i,))

        for path in paths(CONFIG):
            with self.subTest(path=path):
                c = copy.deepcopy(CONFIG)
                parent = c
                for key in path[:-1]:
                    parent = parent[key]
                del parent[path[-1]]
                with self.assertRaisesRegex(ValueError, str(path[-1])):
                    run(c)

    def test_load_config_rejects_missing_and_old_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runtime.yaml'
            c = copy.deepcopy(CONFIG)
            del c['backend']['decode_threshold']
            model = {'active_model': c['backend']['version'], 'backend': c.pop('backend')}
            model_path = Path(directory) / 'model.yaml'
            model_path.write_text(yaml.safe_dump(model))
            c['model_config'] = str(model_path)
            path.write_text(yaml.safe_dump(c))
            with self.assertRaisesRegex(ValueError, 'model.yaml.*backend.decode_threshold'):
                load_config(path)
        c = copy.deepcopy(CONFIG)
        c['decision']['caution_slow'] = .2
        with self.assertRaisesRegex(ValueError, 'obsolete decision keys: caution_slow'):
            validate_runtime(c)

    def test_rule_threshold_action_reason_and_priority_are_configured(self):
        c = copy.deepcopy(CONFIG)
        c['decision']['roi'] = [[0,0], [1,0], [1,1], [0,1]]
        mask = np.zeros((10,10), np.uint8)
        mask[0,0] = 1
        r = dict(mask=mask, boxes=[])
        self.assertEqual(create_processors(c)[1].evaluate(r)['candidate'], 'GO')
        rule = c['decision']['segmentation_rules'][1]
        rule['ratio_gt'] = 0.0
        self.assertEqual(create_processors(c)[1].evaluate(r)['candidate'], 'SLOW')
        rule.update(action='STOP', reason='configured_reason')
        state = create_processors(c)[1].evaluate(r)
        self.assertEqual(state['candidate'], 'STOP')
        self.assertEqual(state['candidate_reasons'], ['configured_reason'])
        c['decision']['action_priority'] = ['SLOW', 'STOP', 'GO']
        r['boxes'] = [([0,0,10,10], .9, 0)]
        self.assertEqual(create_processors(c)[1].evaluate(r)['candidate'], 'SLOW')

    def test_invalid_types_ranges_and_rules(self):
        cases = [('stabilizer', 'enabled', 1), ('stabilizer', 'confirm_n', 1.5),
                 ('geometry', 'pitch_deg', float('nan')), ('geometry', 'height_m', 0),
                 ('backend', 'decode_threshold', True), ('source', 'buffer_size', 0),
                 ('decision', 'stop_frames', 0)]
        for section, key, value in cases:
            with self.subTest(section=section, key=key):
                c = copy.deepcopy(CONFIG)
                c[section][key] = value
                with self.assertRaises(ValueError):
                    validate_runtime(c)
        for key, value in [('ratio_gt', 1.1), ('action', 'DRIVE'), ('class', 'typo')]:
            c = copy.deepcopy(CONFIG)
            c['decision']['segmentation_rules'][0][key] = value
            with self.assertRaises(ValueError):
                validate_runtime(c)

    def test_qos_values_are_passed_without_ros(self):
        qos_module = SimpleNamespace(QoSProfile=Mock(),
            ReliabilityPolicy=SimpleNamespace(RELIABLE='r', BEST_EFFORT='b'),
            HistoryPolicy=SimpleNamespace(KEEP_LAST='last', KEEP_ALL='all'))
        with patch.dict(sys.modules, {'rclpy.qos': qos_module}):
            make_qos(dict(depth=7, reliability='BEST_EFFORT', history='KEEP_ALL'))
        qos_module.QoSProfile.assert_called_once_with(depth=7, reliability='b', history='all')

    def test_publish_subscribe_qos_and_capture_buffer_wiring(self):
        node = Mock()
        ros = SimpleNamespace(init=Mock(), create_node=Mock(return_value=node))
        modules = {'rclpy': ros, 'rclpy.context': SimpleNamespace(Context=Mock()),
                   'rclpy.executors': SimpleNamespace(SingleThreadedExecutor=Mock()),
                   'std_msgs.msg': SimpleNamespace(String=object),
                   'sensor_msgs.msg': SimpleNamespace(Image=object)}
        c = copy.deepcopy(CONFIG)
        for key in ('json', 'udp', 'display', 'video'):
            c['sinks'][key]['enabled'] = False
        c['sinks']['ros2']['enabled'] = True
        for i, key in enumerate(('decision', 'boxes', 'mask')):
            c['sinks']['ros2']['qos'][key]['depth'] = i + 4
        with patch.dict(sys.modules, modules), patch('vision_model.runtime.sinks.make_qos', side_effect=lambda q: dict(q)):
            sink = Sinks(c['sinks'])
        self.assertEqual([call.args[2] for call in node.create_publisher.call_args_list],
                         [c['sinks']['ros2']['qos'][k] for k in ('decision', 'boxes', 'mask')])
        c['source'].update(type='ros2')
        c['source']['ros2_qos'].update(depth=9, reliability='RELIABLE', history='KEEP_ALL')
        with patch.dict(sys.modules, modules), patch('vision_model.runtime.sources.make_qos', side_effect=lambda q: dict(q)):
            source = Source(c['source'])
            source._open()
        self.assertEqual(node.create_subscription.call_args.args[3], c['source']['ros2_qos'])
        c['source'].update(type='v4l2', buffer_size=6)
        capture = Mock()
        with patch('vision_model.runtime.sources.cv2.VideoCapture', return_value=capture):
            Source(c['source'])._open()
        capture.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 6)
        c['source']['type'] = 'csi'
        with patch('vision_model.runtime.sources.cv2.VideoCapture', return_value=capture):
            Source(c['source'])._open()
        self.assertIn('max-buffers=6', capture.open.call_args.args[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
