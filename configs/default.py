from easydict import EasyDict as edict
import yaml

"""
Add default config.
"""
cfg = edict()
cfg.RESUME = None

cfg.MODEL = edict()
cfg.MODEL.TYPE = "TRANSFORMER"
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.BG_PROTOTYPE = True
cfg.MODEL.MAX_CAPACITY = 100
cfg.MODEL.TOTAL_STRIDE = 16
cfg.MODEL.WITH_REFINE = True
cfg.MODEL.REFINE_LAYERS = 2
cfg.MODEL.OBJECT_THRESHOLD = 0.9
cfg.MODEL.DETACH_INTER_FRAME = False
cfg.MODEL.USE_HR_LEVEL_EMBEDDING = True

cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.NAME = "resnet50"
cfg.MODEL.BACKBONE.VERSION = "v1"
cfg.MODEL.BACKBONE.FROZEN = False
cfg.MODEL.BACKBONE.BN_FROZEN = True
  
cfg.MODEL.ATTACH = edict()
cfg.MODEL.ATTACH.USE_SIMILARITY = False
cfg.MODEL.ATTACH.LOCAL_SIMILARITY = False
cfg.MODEL.ATTACH.THRESHOLD_MEAN_FEAT = 0.211
cfg.MODEL.ATTACH.THRESHOLD_SOFT_LABEL = 0.15

cfg.MODEL.TRANSFORMER = edict()
cfg.MODEL.TRANSFORMER.NHEAD = 8
cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 1024
cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
cfg.MODEL.TRANSFORMER.ACTIVATION = 'relu'
cfg.MODEL.TRANSFORMER.REFINE_THRESHOLD = [0.95, 0.9]
cfg.MODEL.TRANSFORMER.DETACH_INTER_LAYER = False
cfg.MODEL.TRANSFORMER.USE_POS_EMBEDDING = True
cfg.MODEL.TRANSFORMER.USING_SCRIBBLE_MEM_FOR_MASK = False

cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.UPSAMPLE_LOGITS = True
cfg.MODEL.DECODER.OUT_DIM = 2
#cfg.MODEL.DECODER.HIDDEN_DIM = 256
cfg.MODEL.DECODER.SHORTCUT_DIMS = [256, 512, 1024]
cfg.MODEL.DECODER.ALIGN_CORNER = True
cfg.MODEL.DECODER.NORM = 'gn'
cfg.MODEL.DECODER.PRED_EDGE = False

cfg.MODEL.SCRIBBLE_PREDICTOR = edict()
cfg.MODEL.SCRIBBLE_PREDICTOR.OUT_DIM = 1
cfg.MODEL.SCRIBBLE_PREDICTOR.LAYERS = 3
cfg.MODEL.SCRIBBLE_PREDICTOR.NORM = 'gn'

cfg.DATASETS = edict()
cfg.DATASETS.MEMORY_BG = True
cfg.DATASETS.MINIMUM_AREA = 100
cfg.DATASETS.ERODE = False
cfg.DATASETS.MAX_OBJECT_NUM = 3
cfg.DATASETS.DILATE_KERNEL_SIZE = 8
cfg.DATASETS.CROP_SIZE = [384, 384]
cfg.DATASETS.SAMPLE_PROBABILITY = [1, 1]
cfg.DATASETS.BIDIRECTION = False
cfg.DATASETS.VIDEO_TEMPORAL_FLIP = False
cfg.DATASETS.DILATE_KERNEL_SIZE = 19

cfg.DATASETS.COCOS = edict()
cfg.DATASETS.COCOS.IMG_DIR = '/datasets/COCO/images'
cfg.DATASETS.COCOS.ANNO_DIR = '/datasets/COCO/scribbles_skeleton_category'
cfg.DATASETS.COCOS.JSON_PATH = '/datasets/COCO/COCO_skeleton_category.json'
cfg.DATASETS.COCOS.RANDOM_FLIP = False
cfg.DATASETS.COCOS.TRANSFORM_MODE = 'perspective'
cfg.DATASETS.COCOS.AFFINE_RANGE= [8,8,8]
cfg.DATASETS.COCOS.CLIP_LENGTH = 4
cfg.DATASETS.COCOS.ADDITIONAL_BG = False

cfg.DATASETS.YTB = edict()
cfg.DATASETS.YTB.ROOT = '/datasets/Youtubu-VOS'
cfg.DATASETS.YTB.CLIP_LENGTH = 4
cfg.DATASETS.YTB.RANDOM_FLIP = False
cfg.DATASETS.YTB.ADDITIONAL_BG = False

cfg.DATASETS.DAVIS = edict()
cfg.DATASETS.DAVIS.PATH = '/datasets/DAVIS'
cfg.DATASETS.DAVIS.CLIP_LENGTH = 4
cfg.DATASETS.DAVIS.RANDOM_FLIP = False
cfg.DATASETS.DAVIS.ADDITIONAL_BG = False

cfg.TRAIN = edict()
cfg.TRAIN.NUM_WORKERS = 8
cfg.TRAIN.PIN_MEMORY = True
cfg.TRAIN.BATCH_SIZE = 4
cfg.TRAIN.MODE = 'clip'
cfg.TRAIN.OPTIMIZER = 'AdamW'# or "Adam"
cfg.TRAIN.EPOCH = 12
cfg.TRAIN.LR = 5e-5
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.2
cfg.TRAIN.DEEP_SUPERVISION = True
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.STEP_SIZE = 6

cfg.TRAIN.MAX_SKIP = 25
cfg.TRAIN.INIT_SKIP = 0
cfg.TRAIN.SKIP_INCREASE_STEP = 1000
cfg.TRAIN.LR_DECREASE_STEP = 1000
cfg.TRAIN.SAMPLES_PER_EPOCH = 10000

cfg.TRAIN.LOSS = edict()
cfg.TRAIN.LOSS.MASK = edict()
cfg.TRAIN.LOSS.MASK.WEIGHT = 2.0
cfg.TRAIN.LOSS.MASK.BG_WEIGHT = 1.0

cfg.TRAIN.LOSS.SMOOTH = edict()
cfg.TRAIN.LOSS.SMOOTH.ALPHA = 10.0
cfg.TRAIN.LOSS.SMOOTH.WITH_SCRIBBLE = False
cfg.TRAIN.LOSS.SMOOTH.WEIGHT = 0.3

cfg.TRAIN.LOSS.CONSISTENCY = edict()
cfg.TRAIN.LOSS.CONSISTENCY.WEIGHT = 0.0

cfg.TRAIN.LOSS.SCRIBBLE_LOSS = edict()
cfg.TRAIN.LOSS.SCRIBBLE_LOSS.USE_FOCAL_LOSS = True
cfg.TRAIN.LOSS.SCRIBBLE_LOSS.ALPHA = 0.25
cfg.TRAIN.LOSS.SCRIBBLE_LOSS.WEIGHT = 1.0

cfg.TEST = edict()
cfg.TEST.USING_INITIAL_BG = True
cfg.TEST.REQUIRE_LAST = False
cfg.TEST.REQUIRE_LAST_SCRIBBLE = False
cfg.TEST.MAX_MEMORY_LENGTH = 20
cfg.TEST.UPDATE_FREQUENCY = 5
cfg.TEST.SCRIBBLE_UPDATE_FREQUENCY = 5


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


