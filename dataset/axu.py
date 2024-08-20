import numpy as np


def uniform_sample_frames(data, n, seed=None):
    """
    Uniformly samples 'n' frames from the dataset 'data' with shape (N, H, W).
    Divides the video into 'n' segments of equal length and randomly samples one frame from each segment.

    Args:
    data: A numpy array with shape (N, H, W), where N is the number of frames.
    n: The number of frames to be sampled, where n < N.
    seed: Optional; a random seed for reproducibility.

    Returns:
    A numpy array with shape (n, H, W) containing the uniformly sampled frames.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility

    N, H, W = data.shape
    segment_length = N / n
    sampled_indices = [
        int(np.random.uniform(i * segment_length, (i + 1) * segment_length))
        for i in range(n)
    ]
    sampled_frames = data[sampled_indices]

    return sampled_frames


def uniform_sample_frames_multi(data_2d, data_3d, confidence, n, seed=None):
    """
    Uniformly samples 'n' frames from the dataset 'data' with shape (B, N, H, W),
    where B is the batch size. Divides the video into 'n' segments of equal length 
    and randomly samples one frame from each segment for every batch identically.

    Args:
    data: A numpy array with shape (B, N, H, W), where N is the number of frames.
    n: The number of frames to be sampled, where n < N for each batch.
    seed: Optional; a random seed for reproducibility.

    Returns:
    A numpy array with shape (B, n, H, W) containing the uniformly sampled frames.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility

    B, N, H, W = data_2d.shape
    segment_length = N / n
    # Generate sampled indices once based on the first batch's frame count
    sampled_indices = [
        int(np.random.uniform(i * segment_length, (i + 1) * segment_length))
        for i in range(n)
    ]
    # Use these indices to sample frames across all batches identically
    confidence = confidence[sampled_indices]
    sampled_frames_3d = data_3d[sampled_indices]
    sampled_frames_2d = data_2d[:, sampled_indices]

    return sampled_frames_2d, sampled_frames_3d, confidence


def expand_to_slow_motion_np(B, F):
    f, K, C = B.shape
    # Create a new frame axis for B_prime
    B_prime = np.zeros((F, K, C), dtype=B.dtype)

    # Calculate the new frame indices (target)
    new_frame_indices = np.linspace(0, f - 1, F)

    # Original frame indices (source)
    original_frame_indices = np.arange(f)

    # Interpolate for each batch, keypoint, and x,y coordinate
    for k in range(K):
        for c in range(C):
            B_prime[:, k, c] = np.interp(
                new_frame_indices, original_frame_indices, B[:, k, c]
            )

    return B_prime