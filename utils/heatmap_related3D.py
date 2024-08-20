# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from utils.axu import scale_and_center_3D

EPS = 1e-3


def generate_a_3d_heatmap(sigma, arr, centers, max_values):
    """Generate a 3D pseudo heatmap for keypoints.

    Args:
        sigma (float): The standard deviation of the Gaussian.
        arr (np.ndarray): The 3D array to store the generated heatmaps. Shape: img_h * img_w * img_d.
        centers (np.ndarray): The 3D coordinates of keypoints. Shape: M * 3.
        max_values (np.ndarray): The max values of each keypoint. Shape: M.

    """
    img_h, img_w, img_d = arr.shape

    for center, max_value in zip(centers, max_values):
        if max_value < EPS:
            continue

        mu_x, mu_y, mu_z = round(center[0]), round(center[1]), round(center[2])
        st_x = max(int(mu_x - 3 * sigma), 0)
        ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
        st_y = max(int(mu_y - 3 * sigma), 0)
        ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
        st_z = max(int(mu_z - 3 * sigma), 0)
        ed_z = min(int(mu_z + 3 * sigma) + 1, img_d)

        x = np.arange(st_x, ed_x, dtype=np.float32)
        y = np.arange(st_y, ed_y, dtype=np.float32)
        z = np.arange(st_z, ed_z, dtype=np.float32)

        if not (len(x) and len(y) and len(z)):
            continue

        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2 + (z - mu_z) ** 2) / (2 * sigma**2))
        patch *= max_value
        arr[st_x:ed_x, st_y:ed_y, st_z:ed_z] = np.maximum(arr[st_x:ed_x, st_y:ed_y, st_z:ed_z], patch)


def fallten_generate_a_limb_heatmap(sigma, arr, starts, ends, start_values, end_values):
    """Generate pseudo heatmap for one limb in one frame.

    Args:
        arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
        starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 2.
        ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 2.
        start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
        end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.

    Returns:
        np.ndarray: The generated pseudo heatmap.
    """

    img_h, img_w = arr.shape
    for start, end, start_value, end_value in zip(
        starts, ends, start_values, end_values
    ):
        value_coeff = min(start_value, end_value)
        if value_coeff < EPS:
            continue

        min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
        min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

        min_x = max(int(min_x - 3 * sigma), 0)
        max_x = min(int(max_x + 3 * sigma) + 1, img_w)
        min_y = max(int(min_y - 3 * sigma), 0)
        max_y = min(int(max_y + 3 * sigma) + 1, img_h)

        x = np.arange(min_x, max_x, 1, np.float32)
        y = np.arange(min_y, max_y, 1, np.float32)

        if not (len(x) and len(y)):
            continue

        y = y[:, None]
        x_0 = np.zeros_like(x)
        y_0 = np.zeros_like(y)

        # distance to start keypoints
        d2_start = (x - start[0]) ** 2 + (y - start[1]) ** 2

        # distance to end keypoints
        d2_end = (x - end[0]) ** 2 + (y - end[1]) ** 2

        # the distance between start and end keypoints.
        d2_ab = (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2

        if d2_ab < 1:
            generate_a_3d_heatmap(sigma, arr, start[None], start_value[None])
            continue

        coeff = (d2_start - d2_end + d2_ab) / 2.0 / d2_ab

        a_dominate = coeff <= 0
        b_dominate = coeff >= 1
        seg_dominate = 1 - a_dominate - b_dominate

        position = np.stack([x + y_0, y + x_0], axis=-1)
        projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
        d2_line = position - projection
        d2_line = d2_line[:, :, 0] ** 2 + d2_line[:, :, 1] ** 2
        d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

        patch = np.exp(-d2_seg / 2.0 / sigma**2)
        patch = patch * value_coeff

        arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)


def flatten_generate_heatmap(data, sigma, num_c, img_h, img_w):
    kps, max_values = data

    arr = np.zeros([num_c, img_h, img_w], dtype=np.float32)

    skeletons = (
        (0, 1),
        (0, 2),
        (0, 3),
        (3, 4),
        (4, 5),
        (0, 9),
        (9, 10),
        (10, 11),
        (2, 6),
        (2, 12),
        (6, 7),
        (7, 8),
        (12, 13),
        (13, 14),
    )

    for i, limb in enumerate(skeletons):
        start_idx, end_idx = limb
        starts = kps[:, start_idx]
        ends = kps[:, end_idx]

        start_values = max_values[:, start_idx]
        end_values = max_values[:, end_idx]
        fallten_generate_a_limb_heatmap(
            sigma, arr[i], starts, ends, start_values, end_values
        )

    return arr


# --------------------------------------------------------------------------------------
class GeneratePoseTarget_3D:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right limbs of skeletons we defined for COCO-17p.
    """

    def __init__(
        self,
        sigma=0.5,
        use_score=False,
        use_gaussian_score=True,
        mean_gaussian_score=0.65,
        scale_gaussian_score=0.16,
        with_kp=True,
        with_limb=False,
        skeletons=(
            (0, 1),
            (0, 2),
            (0, 3),
            (3, 4),
            (4, 5),
            (0, 9),
            (9, 10),
            (10, 11),
            (2, 6),
            (2, 12),
            (6, 7),
            (7, 8),
            (12, 13),
            (13, 14),
        ),
        double=False,
        left_kp=(3, 4, 5, 6, 7, 8),
        right_kp=(9, 10, 11, 12, 13, 14),
        # Not sure what to do with the limbs
        left_limb=(3, 4, 5, 6, 7, 8),
        right_limb=(9, 10, 11, 12, 13, 14),
        heatmap_size=256,
        heatmap_depth=32,
        img_dims=(256, 256, 256),
        scaling=1.0,
        is_training=False,
    ):
        self.sigma = sigma
        self.use_score = use_score
        self.use_gaussian_score = use_gaussian_score
        self.mean_gaussian_score = (mean_gaussian_score,)
        self.scale_gaussian_score = (scale_gaussian_score,)
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.heatmap_size = heatmap_size
        self.heatmap_depth = heatmap_depth
        self.img_dims = img_dims

        assert (
            self.with_kp + self.with_limb == 1
        ), 'One of "with_limb" and "with_kp" should be set as True.'
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.is_training = is_training

    def generate_a_heatmap_3d(self, volume, centers, max_values):
        """Generate pseudo 3D heatmap for one keypoint in one frame.

        Args:
            volume (np.ndarray): The 3D array to store the generated heatmaps. Shape: depth * img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons) in 3D. Shape: M * 3.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo 3D heatmap.
        """
        sigma_x, sigma_y, sigma_z = self.sigma
        depth, img_h, img_w = volume.shape

        for center, max_value in zip(centers, max_values):
            if max_value < EPS:
                continue

            mu_x, mu_y, mu_z = round(center[0]), round(center[1]), round(center[2])
            st_x = max(int(mu_x - 3 * sigma_x), 0)
            ed_x = min(int(mu_x + 3 * sigma_x) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma_y), 0)
            ed_y = min(int(mu_y + 3 * sigma_y) + 1, img_h)
            st_z = max(int(mu_z - 3 * sigma_z), 0)
            ed_z = min(int(mu_z + 3 * sigma_z) + 1, depth)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)
            z = np.arange(st_z, ed_z, 1, np.float32)

            if not (len(x) and len(y) and len(z)):
                continue

            y = y[:, None, None]
            x = x[None, :, None]
            z = z[None, None, :]

            patch = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2) + (z - mu_z) ** 2 / (2 * sigma_z**2)))
            patch = patch * max_value
            volume[st_z:ed_z, st_y:ed_y, st_x:ed_x] = np.maximum(volume[st_z:ed_z, st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap_3d(self, arr, starts, ends, start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame in 3D space.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w * img_d.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: M * 3.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: M * 3.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: M.
        """
        sigma = self.sigma
        img_h, img_w, img_d = arr.shape
        EPS = 1e-5  # Define a small epsilon to avoid division by zero or near-zero distances

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < EPS:
                continue

            # Extend min and max calculation to include z dimension
            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
            min_z, max_z = min(start[2], end[2]), max(start[2], end[2])

            # Adjust bounds for 3D volume
            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)
            min_z = max(int(min_z - 3 * sigma), 0)
            max_z = min(int(max_z + 3 * sigma) + 1, img_d)

            # Create a grid of points within the bounds
            x = np.arange(min_x, max_x, dtype=np.float32)
            y = np.arange(min_y, max_y, dtype=np.float32)
            z = np.arange(min_z, max_z, dtype=np.float32)
            x, y, z = np.meshgrid(x, y, z, indexing='ij')

            # Compute squared distances in 3D
            d2_start = (x - start[0])**2 + (y - start[1])**2 + (z - start[2])**2
            d2_end = (x - end[0])**2 + (y - end[1])**2 + (z - end[2])**2
            d2_ab = (start[0] - end[0])**2 + (start[1] - end[1])**2 + (start[2] - end[2])**2

            if d2_ab < EPS:
                # If start and end are too close, treat this as a single point to avoid division by zero
                continue

            # Compute the projection coefficients in 3D
            coeff = (d2_start - d2_end + d2_ab) / (2.0 * d2_ab)

            # Domination conditions
            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            # Calculate the distance to the line segment in 3D
            projection = start + np.stack([coeff, coeff, coeff], axis=-1) * (end - start)
            d2_line = (x - projection[..., 0])**2 + (y - projection[..., 1])**2 + (z - projection[..., 2])**2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            # Apply Gaussian falloff
            patch = np.exp(-d2_seg / (2.0 * sigma**2)) * value_coeff
            patch = patch * value_coeff
            arr[min_x:max_x, min_y:max_y, min_z:max_z] = np.maximum(arr[min_x:max_x, min_y:max_y, min_z:max_z], patch)

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: kps * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: 1 * kps * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap_3d(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap_3d(
                    arr[i], starts, ends, start_values, end_values
                )

    def gen_an_aug(
        self, results, keypoint_score=None, min_down_scaling=0.1, max_up_scaling=1
    ):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results
        kp_shape = all_kps.shape

        if self.use_score:
            all_kpscores = np.expand_dims(keypoint_score, 0)
        else:
            if self.use_gaussian_score:
                all_kpscores = np.random.normal(
                    loc=self.mean_gaussian_score,
                    scale=self.scale_gaussian_score,
                    size=kp_shape[:-1],
                )
            else:
                all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        all_kpscores = np.clip(all_kpscores, 0, 1)

        img_h, img_w, img_d = self.img_dims

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        img_d = int(img_d * self.scaling + 0.5)
        all_kps[..., :3] *= self.scaling
        kps = all_kps

        aug_kpt = scale_and_center_3D(kps, self.scaling * self.heatmap_size)
        
        # Ensure keypoints are within the heatmap boundaries
        # Replace assertions with clipping to ensure values are within valid range
        aug_kpt = np.clip(aug_kpt, 0, self.heatmap_size - 1)

        num_frame = kp_shape[1]
        new_img_d, new_img_h, new_img_w = self.heatmap_size, self.heatmap_size, self.heatmap_size  # aug_kpt[0].shape
        
        num_c = 0
        if self.with_kp:
            num_c += aug_kpt.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, new_img_d, new_img_h, new_img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = aug_kpt[:, i]
            # M, C
            kpscores = all_kpscores[:, i]

            self.generate_heatmap(ret[i], kps, kpscores)

        return ret.astype(np.float32), aug_kpt[0]

    def __call__(self, results, keypoint_score=None):
        heatmap, kpt = self.gen_an_aug(results, keypoint_score)
        key = "heatmap_imgs" if "imgs" in results else "imgs"

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (
                (self.left_kp, self.right_kp)
                if self.with_kp
                else (self.left_limb, self.right_limb)
            )
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        combined_heatmaps = heatmap.max(axis=1)
        return combined_heatmaps, kpt

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sigma={self.sigma}, "
            f"use_score={self.use_score}, "
            f"with_kp={self.with_kp}, "
            f"with_limb={self.with_limb}, "
            f"skeletons={self.skeletons}, "
            f"double={self.double}, "
            f"left_kp={self.left_kp}, "
            f"right_kp={self.right_kp})"
        )
        return repr_str
