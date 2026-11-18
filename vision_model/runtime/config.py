"""Required runtime settings and validation; never supply tuning defaults."""
import math
from pathlib import Path
import re

import yaml


def require(config, keys, prefix):
    if not isinstance(config, dict):
        raise ValueError(prefix + ' must be a mapping')
    for key in keys:
        if key not in config:
            raise ValueError('missing required setting: ' + prefix + '.' + key)


def number(value, path, minimum=None, maximum=None, integer=False):
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if (not valid_type or not math.isfinite(value) or
            (minimum is not None and value < minimum) or
            (maximum is not None and value > maximum)):
        raise ValueError('invalid numeric setting: ' + path)


def validate_rules(c):
    old = set(c) & {'non_drivable_stop', 'caution_slow', 'obstacle_bottom_margin', 'obstacle_class', 'slow_classes'}
    if old:
        raise ValueError('obsolete decision keys: ' + ', '.join(sorted(old)) +
                         '; migrate to decision.segmentation_rules / box_rules / action_priority')
    require(c, ('segmentation_rules', 'box_rules', 'geometry_rules', 'action_priority',
                'segmentation_classes', 'box_classes'), 'decision')
    priority = c['action_priority']
    if (not isinstance(priority, list) or len(priority) != 3 or
            any(not isinstance(v, str) for v in priority) or set(priority) != {'STOP', 'SLOW', 'GO'}):
        raise ValueError('decision.action_priority must order STOP, SLOW, GO exactly once')
    for key, count in (('segmentation_classes', 3), ('box_classes', 4)):
        classes = c[key]
        if (not isinstance(classes, list) or len(classes) != count or
                any(not isinstance(v, str) or not v for v in classes) or len(set(classes)) != count):
            raise ValueError('decision.' + key + ' must contain unique model class names')
    for kind in ('segmentation', 'box', 'geometry'):
        path = 'decision.' + kind + '_rules'
        rules = c[kind + '_rules']
        if not isinstance(rules, list):
            raise ValueError(path + ' must be a list')
        for i, rule in enumerate(rules):
            p = f'{path}[{i}]'
            fields = ('action', 'reason', 'zone' if kind == 'geometry' else 'class')
            require(rule, fields, p)
            if kind == 'segmentation':
                require(rule, ('ratio_gt',), p)
                number(rule['ratio_gt'], p + '.ratio_gt', 0, 1)
            if kind == 'geometry':
                if rule['zone'] not in ('stop', 'slow', 'none'):
                    raise ValueError(p + '.zone must be stop/slow/none')
            elif rule['class'] not in c['segmentation_classes' if kind == 'segmentation' else 'box_classes']:
                raise ValueError(p + '.class is not a configured model class')
            if any(k in rule for k in ('near_bottom_margin', 'near_action', 'near_reason')):
                if kind != 'box':
                    raise ValueError(p + ' near fields are only valid for box rules')
                require(rule, ('near_bottom_margin', 'near_action', 'near_reason'), p)
                number(rule['near_bottom_margin'], p + '.near_bottom_margin', 0, 1)
                fields = fields + ('near_action', 'near_reason')
            allowed = set(fields) | ({'ratio_gt'} if kind == 'segmentation' else set())
            if kind == 'box' and 'near_bottom_margin' in rule:
                allowed.add('near_bottom_margin')
            if set(rule) - allowed:
                raise ValueError(p + ' unknown rule keys: ' + ', '.join(sorted(set(rule) - allowed)))
            for key in fields:
                if key.endswith('action') and rule[key] not in priority:
                    raise ValueError(p + '.' + key + ' must be a configured action')
                if key.endswith('reason') and (not isinstance(rule[key], str) or not rule[key]):
                    raise ValueError(p + '.' + key + ' must be a nonempty string')


def validate_qos(c, path):
    require(c, ('depth', 'reliability', 'history'), path)
    number(c['depth'], path + '.depth', 1, integer=True)
    if c['reliability'] not in ('RELIABLE', 'BEST_EFFORT'):
        raise ValueError(path + '.reliability must be RELIABLE/BEST_EFFORT')
    if c['history'] not in ('KEEP_LAST', 'KEEP_ALL'):
        raise ValueError(path + '.history must be KEEP_LAST/KEEP_ALL')


def make_qos(c):
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=c['depth'], reliability=getattr(ReliabilityPolicy, c['reliability']),
                      history=getattr(HistoryPolicy, c['history']))


def validate_runtime(c):
    require(c, ('source', 'backend', 'decision', 'geometry', 'stabilizer', 'sinks', 'monitor'), 'runtime')
    source = c['source']
    require(source, ('type', 'uri', 'width', 'height', 'fps', 'rotate', 'speed', 'latest_only',
                     'timeout_s', 'backoff_s', 'backoff_max_s', 'limit', 'csi_sensor', 'topic',
                     'buffer_size', 'ros2_qos'), 'source')
    if source['type'] not in ('v4l2', 'csi', 'rtsp', 'file', 'dir', 'ros2'):
        raise ValueError('unsupported source.type')
    for key in ('width', 'height', 'buffer_size'):
        number(source[key], 'source.' + key, 1, integer=True)
    for key in ('fps', 'timeout_s', 'backoff_s', 'backoff_max_s'):
        number(source[key], 'source.' + key, 0)
        if source[key] == 0:
            raise ValueError('source.' + key + ' must be positive')
    for key in ('limit', 'csi_sensor'):
        number(source[key], 'source.' + key, 0, integer=True)
    number(source['speed'], 'source.speed', .25, 4)
    if type(source['rotate']) is not int or source['rotate'] not in (0, 90, 180, 270):
        raise ValueError('source.rotate must be 0/90/180/270')
    if type(source['latest_only']) is not bool:
        raise ValueError('source.latest_only must be boolean')
    if source['backoff_max_s'] < source['backoff_s']:
        raise ValueError('source.backoff_max_s must be >= source.backoff_s')
    validate_qos(source['ros2_qos'], 'source.ros2_qos')
    validate_backend(c['backend'])
    monitor = c['monitor']
    require(monitor, ('window', 'log_every', 'max_frames'), 'monitor')
    for key in ('window', 'log_every', 'max_frames'):
        number(monitor[key], 'monitor.' + key, 0 if key == 'max_frames' else 1, integer=True)
    sinks = c['sinks']
    require(sinks, ('json', 'udp', 'video', 'display', 'ros2'), 'sinks')
    fields = {'json': ('path',), 'udp': ('host', 'port'), 'video': ('path', 'fps', 'width'),
              'display': ('width', 'seek_seconds'), 'ros2': ('decision_topic', 'mask_topic', 'boxes_topic', 'qos')}
    for kind, keys in fields.items():
        require(sinks[kind], ('enabled',) + keys, 'sinks.' + kind)
        if type(sinks[kind]['enabled']) is not bool:
            raise ValueError('sinks.' + kind + '.enabled must be boolean')
    number(sinks['udp']['port'], 'sinks.udp.port', 1, 65535, integer=True)
    number(sinks['video']['fps'], 'sinks.video.fps', 0)
    if sinks['video']['fps'] == 0:
        raise ValueError('sinks.video.fps must be positive')
    for kind in ('video', 'display'):
        number(sinks[kind]['width'], 'sinks.' + kind + '.width', 1, integer=True)
    seek = sinks['display']['seek_seconds']
    if not isinstance(seek, list) or len(seek) != 2:
        raise ValueError('sinks.display.seek_seconds must contain two positive values')
    for v in seek:
        number(v, 'sinks.display.seek_seconds', 0)
        if v == 0:
            raise ValueError('sinks.display.seek_seconds must be positive')
    require(sinks['ros2']['qos'], ('decision', 'boxes', 'mask'), 'sinks.ros2.qos')
    for kind in ('decision', 'boxes', 'mask'):
        validate_qos(sinks['ros2']['qos'][kind], 'sinks.ros2.qos.' + kind)
    # Pure constructors validate every rule and geometry/stabilizer field before I/O.
    from .stabilizer import create_processors
    create_processors(c)


def validate_backend(b):
    require(b, ('type', 'device', 'provider', 'version', 'threads', 'decode_threshold', 'advice'), 'backend')
    if b['type'] not in ('torch', 'onnx', 'tensorrt'):
        raise ValueError('unsupported backend.type')
    number(b['threads'], 'backend.threads', 1, integer=True)
    number(b['decode_threshold'], 'backend.decode_threshold', 0, 1)
    if type(b['advice']) is not bool:
        raise ValueError('backend.advice must be boolean')


PACKAGE = Path(__file__).resolve().parents[2]


def read_yaml(path):
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('must be a mapping')
        return value
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f'{path}: {exc}') from exc


def load_model_config(path=None):
    """Read AI settings once; relative paths use the package root."""
    path = Path(path) if path is not None else PACKAGE / 'config/model.yaml'
    if not path.is_absolute():
        path = PACKAGE / path
    c = read_yaml(path)
    try:
        require(c, ('active_model', 'backend'), 'model')
        if not isinstance(c['active_model'], str) or not re.fullmatch(
                r'[0-9]{8}_[A-Za-z0-9][A-Za-z0-9_-]*', c['active_model']):
            raise ValueError('invalid setting: active_model (expected YYYYMMDD_name)')
        validate_backend(c['backend'])
        version = c['backend']['version']
        if version is not None and (not isinstance(version, str) or not re.fullmatch(
                r'[0-9]{8}_[A-Za-z0-9][A-Za-z0-9_-]*', version)):
            raise ValueError('invalid setting: backend.version (expected null or YYYYMMDD_name)')
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return c


def load_config(path):
    """Merge two files into the existing runtime dictionary before any I/O."""
    path = Path(path)
    runtime = read_yaml(path)
    if 'backend' in runtime or 'active_model' in runtime:
        raise ValueError(f'{path}: obsolete runtime settings; move backend and active_model to config/model.yaml; '
                         'set model_config: config/model.yaml (package-root relative)')
    try:
        require(runtime, ('model_config',), 'runtime')
        if not isinstance(runtime['model_config'], str) or not runtime['model_config']:
            raise ValueError('model_config must be a nonempty path')
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    model = load_model_config(runtime.pop('model_config'))
    runtime['backend'] = dict(model['backend'])
    if runtime['backend']['version'] is None:
        runtime['backend']['version'] = model['active_model']
    try:
        validate_runtime(runtime)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return runtime
