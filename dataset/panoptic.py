from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import torch
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np

np.set_printoptions(suppress=True, precision=10)
import json_tricks as json
import pickle
import logging
import os
import copy
from tqdm import tqdm
import pandas as pd

from utils.transforms import projectPoints
from utils.heatmap_related import GeneratePoseTarget, CV2BasedLimbGenerated
from utils.heatmap_related3D import GeneratePoseTarget_3D
from dataset.axu import uniform_sample_frames, uniform_sample_frames_multi

logger = logging.getLogger(__name__)

TRAIN_LIST = [
    "160224_haggling1",
    "160226_haggling1",
    "170221_haggling_b1",
    "170221_haggling_b3",
    "170221_haggling_m1",
    "170221_haggling_m2",
    "170221_haggling_m3",
    "170224_haggling_a2",
    "170224_haggling_a3",
    "170224_haggling_b1",
    "170224_haggling_b2",
    "170224_haggling_b3",
    "170228_haggling_a1",
    "170228_haggling_a2",
    "170228_haggling_b1",
    "170228_haggling_b2",
    "170228_haggling_b3",
    "170404_haggling_a1",
    "170404_haggling_a3",
    "170404_haggling_b1",
    "170404_haggling_b2",
    "170404_haggling_b3",
    "170407_haggling_a2",
    "170407_haggling_a3",
    "170407_haggling_b1",
    "170407_haggling_b2",
    "160422_ultimatum1",
    "160906_band1",
    "160906_band2",
    "160906_ian2",
    "160906_ian3",
    "160906_ian5",
    "160906_pizza1",
    "161029_flute1",
    "161029_piano1",
    "161029_piano2",
    "161029_piano4",
    "170307_dance5",
    "170407_office2",
    "171026_cello3",
    "171026_pose1",
    "171026_pose3",
    "171204_pose1",
    "171204_pose2",
    "171204_pose3",
    "171204_pose4",
    "171204_pose6",
]

VALIDATION_LIST = [
    "170407_haggling_b3",
    "170407_haggling_a1",
    "170404_haggling_a2",
    "170228_haggling_a3",
    "170224_haggling_a1",
    "170221_haggling_b2",
    "160422_haggling1",
    "160906_band3",
    "160906_ian1",
    "161029_piano3",
    "170915_office1",
    "171026_pose2",
    "171204_pose5",
]

CAMERA_LIST_EXO = [1, 2, 4, 6, 7, 10, 13, 17, 19, 28]
CAMERA_LIST_EGO = [0, 3, 5, 8, 9, 14, 15, 23, 24, 25]
CAMERA_LIST_EXO_TRIPLES = [
    ["00_01", "00_04", "00_07"],
    ["00_17", "00_28", "00_13"],
    ["00_02", "00_06", "00_19"],
    ["00_10", "00_17", "00_28"],
]

JOINTS_DEF = {
    "neck": 0,
    "nose": 1,
    "mid-hip": 2,
    "l-shoulder": 3,
    "l-elbow": 4,
    "l-wrist": 5,
    "l-hip": 6,
    "l-knee": 7,
    "l-ankle": 8,
    "r-shoulder": 9,
    "r-elbow": 10,
    "r-wrist": 11,
    "r-hip": 12,
    "r-knee": 13,
    "r-ankle": 14,
    "l-eye": 15,
    "l-ear": 16,
    "r-eye": 17,
    "r-ear": 18,
    # "l-hand": 19,
    # "l-hand-tip": 20,
    # "l-thumb": 21,
    # "r-hand": 22,
    # "r-hand-tip": 23,
    # "r-thumb": 24,
}
HAND_JOINTS = {
    "hand": 9,
    "hand-tip": 12,
    "thumb": 4,
}

JOINTS_PAIRS = [
    ("nose", "l-shoulder"),
    ("nose", "r-shoulder"),
    ("l-shoulder", "l-elbow"),
    ("l-elbow", "l-wrist"),
    ("r-shoulder", "r-elbow"),
    ("r-elbow", "r-wrist"),
    ("l-shoulder", "l-hip"),
    ("l-hip", "l-knee"),
    ("l-knee", "l-ankle"),
    ("r-shoulder", "r-hip"),
    ("r-hip", "r-knee"),
    ("r-knee", "r-ankle"),
    ("l-eye", "nose"),
    ("r-eye", "nose"),
    ("l-eye", "l-ear"),
    ("r-eye", "r-ear"),
    ("r-hip", "l-hip"),
    # ("neck", "nose"),
    # ("neck", "mid-hip"),
    # ("mid-hip", "l-hip"),
    # ("mid-hip", "r-hip"),
    # ("l-wrist", "l-hand"),
    # ("l-hand", "l-hand-tip"),
    # ("l-hand", "l-thumb"),
    # ("r-wrist", "r-hand"),
    # ("r-hand", "r-hand-tip"),
    # ("r-hand", "r-thumb"),
]


SKELETON = [(JOINTS_DEF[joint1], JOINTS_DEF[joint2]) for joint1, joint2 in JOINTS_PAIRS]

LEFT_LIMB = (3, 4, 5, 6, 7, 8)
RIGHT_LIMB = (9, 10, 11, 12, 13, 14)


class Panoptic:
    def __init__(
        self,
        root="",
        stride=15,
        joint_req=0.9,
        camera_num=10,
        data_size=300,
        window_size=48,
        frame_interval=1,
        perspective_ego=False,
        resolution=[1920, 1080],
        is_training=True,
        hm2d=True,
        hm3d=True,
        num_camera_selection=3,
        max_combine=True,
        heatmap_generator={},
        **kwargs,
    ):

        this_dir = os.path.dirname(__file__)
        self.dataset_root = root
        if isinstance(root, str):
            dataset_root = os.path.join(this_dir, "../..", root)
            self.dataset_root = os.path.abspath(dataset_root)

        self.stride = stride
        self.joint_req = joint_req
        self.image_set = "train" if is_training else "validation"
        self.num_views = camera_num
        self.resolution = resolution
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.num_views = camera_num
        self.data_size = data_size
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.total_window = self.window_size * self.frame_interval
        self.db = []
        self.is_training = is_training
        self.hm2d = hm2d
        self.hm3d = hm3d
        self.num_camera_selection = num_camera_selection

        self.joints_def = JOINTS_DEF
        self.hand_indices = list(HAND_JOINTS.values())
        self.joint_indices = list(JOINTS_DEF.values())
        self.heatmap_generator_3D = self.heatmap_generator_2D = None

        if hm2d:
            self.heatmap_generator_2D = GeneratePoseTarget(
                **heatmap_generator,
                skeletons=SKELETON,
                left_kp=LEFT_LIMB,
                left_limb=LEFT_LIMB,
                right_kp=RIGHT_LIMB,
                right_limb=RIGHT_LIMB,
                is_training=is_training,
                max_combine=max_combine,
            )
        if hm3d:
            self.heatmap_generator_3D = GeneratePoseTarget_3D(
                **heatmap_generator,
                skeletons=SKELETON,
                left_kp=LEFT_LIMB,
                left_limb=LEFT_LIMB,
                right_kp=RIGHT_LIMB,
                right_limb=RIGHT_LIMB,
                is_training=is_training,
            )
        self.differences = {}
        self.multi_cam = 4

        self.sequence_list = eval(self.image_set.upper() + "_LIST")
        self.cam_list = CAMERA_LIST_EGO if perspective_ego else CAMERA_LIST_EXO
        self.cam_list = [(0, i) for i in self.cam_list]
        self.cam_type = "ego" if perspective_ego else "exo"
        self.num_views = len(self.cam_list)

        self.db_file = (
            f"3D_group_{self.image_set}_cam{self.num_views}{self.cam_type}_300.pkl"
        )
        # print(f"Using {self.db_file}")
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if osp.exists(self.db_file):
            info = pickle.load(open(self.db_file, "rb"))
            assert info["sequence_list"] == self.sequence_list
            assert info["cam_list"] == self.cam_list
            self.vf = info["valid_frames"]
            self.db = info["data"]
            self.db3d = info["data3D"]
            self.meta = info["meta"]
            self.meta_3D = info["meta3D"]
            self.differences = info["dist"]
        else:
            self.vf, self.db, self.db3d, self.meta, self.meta_3D, _ = self._get_db()
            info = {
                "sequence_list": self.sequence_list,
                "cam_list": self.cam_list,
                "valid_frames": self.vf,
                "data": self.db,
                "data3D": self.db3d,
                "meta": self.meta,
                "meta3D": self.meta_3D,
                "dist": self.differences,
            }
            pickle.dump(info, open(self.db_file, "wb"))

        self.vf_size = len(self.vf)

    def _get_db(self):
        width = 1920
        height = 1080
        db = []
        db_3D = {}
        for seq in tqdm(self.sequence_list, desc="Sequences", position=0):
            cameras = self._get_cam(seq)

            curr_body_anno = osp.join(self.dataset_root, seq, "hdPose3d_stage1_coco19")
            anno_body_files = sorted(glob.iglob("{:s}/*.json".format(curr_body_anno)))

            prev_pose3d = {}

            for i, b_file in tqdm(
                enumerate(anno_body_files),
                desc=f"Files in {seq}",
                total=len(anno_body_files),
                position=1,
                leave=False,
            ):
                try:
                    with open(b_file) as dfile:
                        bodies = json.load(dfile)["bodies"]

                except:
                    print(b_file)
                    continue

                if len(bodies) == 0:
                    continue

                for k, v in cameras.items():
                    postfix = osp.basename(b_file).replace("body3DScene", "")
                    prefix = "{:02d}_{:02d}".format(k[0], k[1])

                    for body in bodies:

                        frame = postfix[1:-5]
                        full_id = f"{prefix}{body['id']}"
                        pose3d = np.array(body["joints19"]).reshape((-1, 4))

                        confidence = pose3d[:, -1]
                        joints_vis = confidence > 0.1

                        # pose3d = np.concatenate((pose3d[:, 0:3], two_hands[:, 0:3]))
                        pose3d = pose3d[:, :-1][self.joint_indices]

                        # Coordinate transformation
                        M = np.array(
                            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
                        )
                        pose3d_dot = pose3d.dot(M)

                        pose2d = np.zeros((pose3d_dot.shape[0], 2))
                        pose2d[:, :2] = projectPoints(
                            pose3d_dot.transpose(),
                            v["K"],
                            v["R"],
                            v["t"],
                            v["distCoef"],
                        ).transpose()[:, :2]

                        x_check = np.bitwise_and(
                            pose2d[: len(joints_vis), 0] >= 0,
                            pose2d[: len(joints_vis), 0] <= width - 1,
                        )
                        y_check = np.bitwise_and(
                            pose2d[: len(joints_vis), 1] >= 0,
                            pose2d[: len(joints_vis), 1] <= height - 1,
                        )
                        check = np.bitwise_and(x_check, y_check)
                        joints_vis[np.logical_not(check)] = 0
                        vis_perc = np.sum(joints_vis) / len(joints_vis)

                        if vis_perc <= self.joint_req:
                            continue

                        if seq not in db_3D.keys():
                            db_3D[seq] = {}
                        if body["id"] not in db_3D[seq].keys():
                            db_3D[seq][body["id"]] = {}
                        if frame not in db_3D[seq][body["id"]].keys():
                            db_3D[seq][body["id"]][frame] = {
                                "pose": pose3d,
                                "cameras": 1,
                            }
                        else:
                            db_3D[seq][body["id"]][frame]["cameras"] += 1

                        if body["id"] not in prev_pose3d.keys():
                            prev_pose3d[body["id"]] = copy.deepcopy(pose3d)

                        difference = int(
                            max(
                                np.linalg.norm(pose3d - prev_pose3d[body["id"]], axis=1)
                            )
                        )
                        self.differences[difference] = (
                            self.differences.get(difference, 0) + 1
                        )

                        prev_pose3d[body["id"]] = copy.deepcopy(pose3d)

                        if len(pose2d) > 0:
                            db.append(
                                {
                                    "frame": frame,
                                    "video": seq,
                                    "joints_2d": pose2d,
                                    "conf": confidence,
                                    "camera": prefix,
                                    "id": body["id"],
                                }
                            )
        self.show_joint_delta_distribution()

        db = sorted(db, key=lambda x: (x["video"], x["id"], int(x["camera"])))

        valid_frames = []
        removal = []
        meta_3D = []
        all_3D_poses = []
        start = len(valid_frames)
        for seq, bodies in db_3D.items():
            for body_id, frames in bodies.items():
                length = len(frames)
                range_of = list(range(start, start + length - self.window_size))
                valid_frames.extend(range_of)
                start += length

                for frame_id, frame in frames.items():
                    all_3D_poses.append(frame["pose"])
                    if frame["cameras"] < 4:
                        removal.append(len(all_3D_poses) - 1)

                    # Append the data for the current (seq, body_id) to the list
                    meta_3D.append(
                        {
                            "video": seq,
                            "id": body_id,
                            "frame": frame_id,
                            "cameras": frame["cameras"],
                        }
                    )

        valid_frames = [x for x in valid_frames if x not in removal]

        meta_db = pd.DataFrame(
            [{k: v for k, v in d.items() if k != "joints_2d"} for d in db]
        )

        all_3D_poses = np.array(all_3D_poses)
        meta_3D = pd.DataFrame(meta_3D)
        valid_frames = np.array(valid_frames)
        skel_array = np.array([i["joints_2d"] for i in db])
        unique_combinations = meta_db[["video", "camera", "id"]].drop_duplicates()

        return (
            valid_frames,
            skel_array,
            all_3D_poses,
            meta_db,
            meta_3D,
            unique_combinations,
        )

    def show_joint_delta_distribution(
        self,
    ):
        # Extract keys and values

        for i in range(50):
            self.differences[i] = 0

        x = list(self.differences.keys())
        y = list(self.differences.values())

        # Create a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)

        # Labeling the axes
        plt.xlabel("Integer Values")
        plt.ylabel("Frequency")
        plt.title("Logarithmic Frequency Distribution of Integer Values")

        # Set y-axis to logarithmic scale
        plt.yscale("log")

        # Save the plot to a file
        plt.savefig(f"logarithmic_frequency_distribution_{self.image_set}.png")

        # Optionally display the plot in the notebook (if you're using one)
        # plt.show()

        print("Plot saved as 'logarithmic_frequency_distribution.png'")

    def _get_cam(self, seq):
        cam_file = osp.join(self.dataset_root, seq, "calibration_{:s}.json".format(seq))
        with open(cam_file) as cfile:
            calib = json.load(cfile)

        M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        cameras = {}
        for cam in calib["cameras"]:
            if (cam["panel"], cam["node"]) in self.cam_list:
                sel_cam = {}
                sel_cam["K"] = np.array(cam["K"])
                sel_cam["distCoef"] = np.array(cam["distCoef"])
                sel_cam["R"] = np.array(cam["R"]).dot(M)
                sel_cam["t"] = np.array(cam["t"]).reshape((3, 1))
                cameras[(cam["panel"], cam["node"])] = sel_cam
        return cameras

    def compress_3d_heatmap(self, heatmap):
        # Permute to change the dimensions to (depth, height, width, channels)
        percep_gt = heatmap.permute(1, 2, 3, 0)

        # Max across each axis
        max_x = torch.max(percep_gt, dim=1)[0]  # Max across x axis
        max_y = torch.max(percep_gt, dim=2)[0]  # Max across y axis
        max_z = torch.max(percep_gt, dim=3)[0]  # Max across z axis

        # Stack them together
        heatmap_xyz = torch.stack(
            [max_x, max_y, max_z], dim=0
        )  # (3, depth, height, width)

        return heatmap_xyz

    # def expand_compressed_3d_heatmap(self, compressed_heatmap, original_shape):
    #     # Initialize an empty tensor for the expanded heatmap
    #     expanded_heatmap = torch.zeros(original_shape)

    #     # Re-expand the heatmap to original dimensions
    #     expanded_heatmap = expanded_heatmap.permute(1, 2, 3, 0)
    #     expanded_heatmap[:, :, :, 0] = compressed_heatmap[0].unsqueeze(3).expand(original_shape[1], original_shape[2], original_shape[3], original_shape[0])
    #     expanded_heatmap[:, :, :, 1] = compressed_heatmap[1].unsqueeze(2).expand(original_shape[1], original_shape[2], original_shape[3], original_shape[0])
    #     expanded_heatmap[:, :, :, 2] = compressed_heatmap[2].unsqueeze(1).expand(original_shape[1], original_shape[2], original_shape[3], original_shape[0])
    #     expanded_heatmap = expanded_heatmap.permute(3, 0, 1, 2)

    #     return expanded_heatmap

    def calculate_rotation_angle(self, data_3D):
        l_shoulder = data_3D[JOINTS_DEF["l-shoulder"]]
        r_shoulder = data_3D[JOINTS_DEF["r-shoulder"]]
        shoulder_vector = r_shoulder - l_shoulder
        angle = np.arctan2(shoulder_vector[2], shoulder_vector[0])
        return angle

    def straighten_by_initial_frame(self, data_3D):
        def rotate_skeleton(skeleton, rotation_matrix):
            return np.dot(skeleton, rotation_matrix)

        def get_rotation_matrix_y(angle):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])

        initial_frame = data_3D[0]
        angle = self.calculate_rotation_angle(initial_frame)
        rotation_matrix = get_rotation_matrix_y(-angle)

        rotated_data_3D = np.array(
            [rotate_skeleton(frame, rotation_matrix) for frame in data_3D]
        )

        return rotated_data_3D

    def __len__(self):
        return self.vf_size // self.stride

    def __getitem__(self, index):
        idx = self.vf[:: self.stride][index]
        data_3D = self.db3d[idx : idx + self.data_size]
        meta = self.meta_3D.iloc[idx]

        condition1 = self.meta["video"].str.contains(meta["video"])
        condition2 = self.meta["id"] == meta["id"]
        condition3 = self.meta["frame"] == meta["frame"]
        fr = self.meta[condition1 & condition2 & condition3]
        # fr = fr.iloc[:self.num_camera_selection]
        fr = pd.concat(
            [fr.iloc[:].sample(n=self.num_camera_selection)]
        )  # fr.iloc[[0]],
        # fr = fr.iloc[:].sample(n=1)

        data_2D = np.zeros((len(fr), self.data_size, len(self.joints_def), 2))
        try:
            for i, (index, row) in enumerate(fr.iterrows()):
                data_2D[i] = self.db[index : index + self.data_size]
        except:
            data_2D = np.zeros((len(fr), self.data_size, len(self.joints_def), 2))

        confidence = np.array(
            self.meta.iloc[idx : idx + self.data_size]["conf"].to_list()
        )

        # end_idx = 64
        end_idx = np.random.randint(16, high=data_3D.shape[0])

        data_2D = data_2D[:, :end_idx]
        data_3D = data_3D[:end_idx]
        confidence = confidence[:end_idx]

        data_2D, data_3D, confidence = uniform_sample_frames_multi(
            data_2D, data_3D, confidence, self.window_size
        )

        if self.hm2d and self.hm3d:
            data_3D = self.straighten_by_initial_frame(data_3D=data_3D)

        # heatmap_2D = np.zeros((self.window_size, 3, self.heatmap_generator_2D.heatmap_size, self.heatmap_generator_2D.heatmap_size), dtype=np.float32)
        heatmap_2D = heatmap_3D = torch.empty(0)
        if self.heatmap_generator_2D is not None:
            heatmap_2D, kpts = self.heatmap_generator_2D(data_2D, confidence)
            heatmap_2D = torch.tensor(heatmap_2D)

        if self.heatmap_generator_3D is not None:
            heatmap_3D, kpts = self.heatmap_generator_3D(
                np.expand_dims(data_3D, axis=0), confidence
            )
            heatmap_3D = torch.tensor(heatmap_3D).permute(3, 0, 1, 2)
            heatmap_3D = self.compress_3d_heatmap(heatmap_3D)

        # new_heatmap_3D = self.expand_compressed_3d_heatmap(compressed_heatmap_3D, original_shape=heatmap_3D.shape)

        heatmap_2D = torch.nan_to_num(heatmap_2D, nan=0.0)
        heatmap_3D = torch.nan_to_num(heatmap_3D, nan=0.0)

        return heatmap_2D, heatmap_3D
