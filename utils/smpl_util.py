class BODY25JointMapper:
    SMPL_TO_BODY25 = [
        24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34
    ]

    def __init__(self):
        self.mapping = self.SMPL_TO_BODY25

    def __call__(self, smpl_output, *args, **kwargs):
        return smpl_output.joints[:, self.mapping]
    
class SMPL2CUSTOMJointMapper:
    """
    CUSTOM = COCO + NECK
    """
    SMPL_TO_CUSTOM = [
        24, 12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28
    ]

    def __init__(self):
        self.mapping = self.SMPL_TO_CUSTOM

    def __call__(self, smpl_output, *args, **kwargs):
        return smpl_output.joints[:, self.mapping]


HEATMAP_THRES = 0.30
PAF_THRES = 0.05
PAF_RATIO_THRES = 0.95
NUM_SAMPLE = 10
MIN_POSE_JOINT_COUNT = 4
MIN_POSE_LIMB_SCORE = 0.4
NUM_JOINTS = 25
BODY25_POSE_INDEX = [(0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25),
                     (26, 27), (6, 7), (2, 3), (4, 5), (8, 9), (10, 11),
                     (12, 13), (30, 31), (32, 33), (36, 37), (34, 35),
                     (38, 39), (20, 21), (28, 29), (40, 41), (42, 43),
                     (44, 45), (46, 47), (48, 49), (50, 51)]
BODY25_PART_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                     (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
                     (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (2, 17),
                     (5, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23),
                     (11, 24)]

joints_name = [
    "Nose",         # 0  => 24 (SMPL)
    "Neck",         # 1  => 12
    "RShoulder",    # 2  => 17
    "RElbow",       # 3  => 19
    "RWrist",       # 4  => 21
    "LShoulder",    # 5  => 16
    "LElbow",       # 6  => 18
    "LWrist",       # 7  => 20
    "MidHip",       # 8  => 0
    "RHip",         # 9  => 2
    "RKnee",        # 10 => 5
    "RAnkle",       # 11 => 8
    "LHip",         # 12 => 1
    "LKnee",        # 13 => 4
    "LAnkle",       # 14 => 7
    "REye",         # 15 => 25
    "LEye",         # 16 => 26
    "REar",         # 17 => 27
    "LEar",         # 18 => 28
    "LBigToe",      # 19 => 29
    "LSmallToe",    # 20 => 30
    "LHeel",        # 21 => 31
    "RBigToe",      # 22 => 32
    "RSmallToe",    # 23 => 33
    "RHeel",        # 24 => 34
]

# OPENPOSE
# SELECT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 
#                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# CUSTOM
SELECT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                 13, 14, 15, 16, 17]