import numpy as np
import cv2
import os
import torch
import torch.distributed as dist
from scipy import interpolate
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def scale_center_per_frame(keypoints_frames: list[Keypoint], N: int):
    scaled_and_centered_frames = []

    for frame_kps in keypoints_frames:
        # Calculate the bounding box for the current frame's keypoints
        min_x = min(kp.x for kp in frame_kps)
        max_x = max(kp.x for kp in frame_kps)
        min_y = min(kp.y for kp in frame_kps)
        max_y = max(kp.y for kp in frame_kps)

        # Determine width and height
        width = max_x - min_x
        height = max_y - min_y

        if width == 0 and height == 0:
            scaled_and_centered_frames.append(frame_kps)
            continue

        # Calculate scale factor
        scale_factor = min(N / width, N / height)

        # Calculate the new scaled dimensions
        scaled_width = width * scale_factor
        scaled_height = height * scale_factor

        # Calculate offsets for centering
        offset_x = (N - scaled_width) / 2 - min_x * scale_factor
        offset_y = (N - scaled_height) / 2 - min_y * scale_factor

        # Apply scaling and centering
        augmenter = iaa.Sequential(
            [
                iaa.Affine(scale=scale_factor),  # Scale
                iaa.Affine(
                    translate_px={"x": round(offset_x), "y": round(offset_y)}
                ),  # Center
            ]
        )

        koi = KeypointsOnImage(frame_kps, shape=(N, N))
        koi_aug = augmenter(keypoints=koi)

        scaled_and_centered_frames.append(koi_aug.keypoints)

    return scaled_and_centered_frames


def resize_keypoints(keypoints, original_size=(1920, 1080), target_size=(128, 128)):
    """
    Resizes keypoints from an original resolution to a target resolution
    while maintaining the aspect ratio.

    Parameters:
    - keypoints: np.array of shape (N, F, J, 2) containing keypoints for N batches,
      F frames, J keypoints, and 2 for (x, y) coordinates.
    - original_size: Tuple specifying the original resolution (width, height).
    - target_size: Tuple specifying the target resolution (width, height).

    Returns:
    - resized_keypoints: np.array of the resized keypoints maintaining the aspect ratio.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size

    # Calculate the scaling factors for width and height separately
    scale_width = target_width / original_width
    scale_height = target_height / original_height

    # Choose the smaller scale factor to maintain aspect ratio
    scale = min(scale_width, scale_height)

    # Resize keypoints
    resized_keypoints = keypoints * scale

    # If you need to center the keypoints in the new resolution, calculate the offset
    new_width = original_width * scale
    new_height = original_height * scale
    offset_x = (target_width - new_width) / 2 if new_width < target_width else 0
    offset_y = (target_height - new_height) / 2 if new_height < target_height else 0

    # Apply offset to center the keypoints
    resized_keypoints[..., 0] += offset_x
    resized_keypoints[..., 1] += offset_y

    return resized_keypoints


def scale(keypoints_frames: list[Keypoint], N: int) -> list[Keypoint]:
    # Calculate global bounding box across all frames
    global_min_x = min(min(kp.x for kp in frame) for frame in keypoints_frames)
    global_max_x = max(max(kp.x for kp in frame) for frame in keypoints_frames)
    global_min_y = min(min(kp.y for kp in frame) for frame in keypoints_frames)
    global_max_y = max(max(kp.y for kp in frame) for frame in keypoints_frames)

    # Determine global width and height
    global_width = global_max_x - global_min_x
    global_height = global_max_y - global_min_y

    # Calculate scale factor to fit the global bounding box within N x N frame
    scale_factor = min(N / global_width, N / global_height)

    # Apply scaling to all frames using the calculated scale factor
    scaled_frames = []
    for frame_kps in keypoints_frames:
        # Apply scaling
        augmenter = iaa.Affine(scale=scale_factor)
        koi = KeypointsOnImage(frame_kps, shape=(N, N))
        koi_aug = augmenter(keypoints=koi)

        # Optionally ensure keypoints are within the N x N frame (might not be necessary)
        for kp in koi_aug.keypoints:
            kp.x = min(max(kp.x, 0), N - 1)
            kp.y = min(max(kp.y, 0), N - 1)

        scaled_frames.append(koi_aug.keypoints)

    return scaled_frames


def scale_and_center(kps: list[Keypoint], N: int) -> list[Keypoint]:
    # Calculate global bounding box
    for frame_kps in kps:
        x_coordinates = [kp.x for kp in frame_kps]
        y_coordinates = [kp.y for kp in frame_kps]
        min_x, max_x = min(x_coordinates), max(x_coordinates)
        min_y, max_y = min(y_coordinates), max(y_coordinates)

    # Scale factor and skeleton dimensions
    scale_factor = min(N / (max_x - min_x), N / (max_y - min_y))
    scaled_width = (max_x - min_x) * scale_factor
    scaled_height = (max_y - min_y) * scale_factor

    # Offsets for centering
    offset_x = (N - scaled_width) / 2 - (min_x * scale_factor)
    offset_y = (N - scaled_height) / 2 - (min_y * scale_factor)
    # Create an augmenter for scaling
    augmenter = iaa.Sequential(
        [
            iaa.Affine(scale=scale_factor),  # Scale keypoints
            iaa.Affine(
                translate_px={
                    "x": round(offset_x),
                    "y": round(offset_y),
                }
            ),  # Center keypoints
        ]
    )

    # Apply the augmentation to all frames
    aug_kpt = []
    for frame_kps in kps:
        koi = KeypointsOnImage(frame_kps, shape=(N, N))  # Adjust the shape as needed
        koi_aug = augmenter(
            keypoints=koi,
        )
        aug_kpt.append(koi_aug.keypoints)

    return aug_kpt

def scale_and_center_3D(kps: np.ndarray, N: int) -> np.ndarray:
    """
    Scale and center 3D keypoints.

    Args:
    kps: A numpy array of shape (1, num_frames, num_keypoints, 3) for x, y, z coordinates.
    N: The size to which we want to scale the smallest dimension of the bounding box.

    Returns:
    A numpy array of scaled and centered keypoints.
    """

    # Calculate global bounding box
    min_coords = np.min(kps, axis=(1, 2))
    max_coords = np.max(kps, axis=(1, 2))
    
    # Scale factors for each dimension
    scale_factors = N / (max_coords - min_coords)
    scale_factor = np.min(scale_factors) * 0.9
    
    scaled_kps = kps * scale_factor
    min_coords *= scale_factor
    max_coords *= scale_factor
    
    # Offsets for centering in x, y, and z
    offsets = (N - (max_coords - min_coords)) / 2 - min_coords
    offsets = np.expand_dims(offsets, axis=(1,2))
    
    # Scale and center
    scaled_and_centered_kps = scaled_kps + offsets
    
    return scaled_and_centered_kps


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_state_dict(
    model, state_dict, prefix="", ignore_missing="relative_position_index"
):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split("|"):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(ignore_missing_keys) > 0:
        print(
            "Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys
            )
        )
    if len(error_msgs) > 0:
        print("\n".join(error_msgs))


def fill_the_model(model, args):

    if args.config.Pretrained_Models.pretrain_saved_file == "":
        print("XX> Pretrained model not found.")
        return

    checkpoint = torch.load(
        args.config.Pretrained_Models.pretrain_saved_file, map_location="cpu"
    )["weights"]

    print("Load ckpt from %s" % args.config.Pretrained_Models.pretrain_saved_file)
    checkpoint_model = None
    for model_key in args.model_key.split("|"):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if (
        model.use_rel_pos_bias
        and "rel_pos_bias.relative_position_bias_table" in checkpoint_model
    ):
        print(
            "Expand the shared relative position embedding to each transformer block. "
        )
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = (
                rel_pos_bias.clone()
            )

        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1
            )
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print(
                    "Position interpolate for %s from %dx%d to %dx%d"
                    % (key, src_size, src_size, dst_size, dst_size)
                )
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions = %s" % str(x))
                print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(rel_pos_bias.device)
                    )

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    # interpolate position embedding
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def convert_hm_to_rgb(array, offset=10):
    """
    Convert a 3D numpy array (T, H, W) to a 3D RGB representation (T, 3, H, W).

    Parameters:
    array (numpy.ndarray): Input array of shape (N, T, H, W).
    colormap (str): The name of the colormap to use.

    Returns:
    numpy.ndarray: Output array of shape (T, 3, H, N * W).
    """
    T, H, W = array.shape
    output = np.zeros((T, 3, H, W), dtype=np.uint8)

    # Apply colormap to each time step
    for t in range(T):
        # Normalize the heatmap for display

        normalized_heatmap = np.clip(array[t], a_min=0.0, a_max=1.0) * 255

        colored_heatmap = cv2.applyColorMap(
            normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
        )

        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        colored_heatmap = colored_heatmap.transpose((2, 0, 1))

        output[t] = colored_heatmap

    return output


if __name__ == "__main__":
    # Example usage
    N, T, H, W = 4, 60, 256, 256  # dimensions
    example_array = np.zeros((N, T, H, W))
    for n in range(1, N):
        example_array[n] = np.ones((T, H, W)) * (n / N)  # Gaussian distribution

    rgb_array = convert_hm_to_rgb(example_array).transpose((0, 3, 2, 1))

    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    _, new_W, new_H, _ = rgb_array.shape

    out = cv2.VideoWriter(
        "test_rgb.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 30, (new_W, new_H)
    )

    for i in range(T):
        # Write to video
        out.write(rgb_array[i])

    out.release()
