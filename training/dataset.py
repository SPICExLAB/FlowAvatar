"""Datasets over the preprocessed AMASS sequences (see training/prepare_data.py).

Every sample is a dict of float32 tensors::

    input         [T, 90]     sparse tracking features
    local_pose    [T, 132]    local joint rotations, 22 x 6D, root first
    global_pose   [T, 132]    global joint rotations, 22 x 6D
    joints        [T, 22, 3]  world-space joint positions
    betas         [T, 10]     SMPL-X shape parameters
    foot_contact  [T, 22]     1 at ankle/foot joints in ground contact
    floor_height  []          estimated floor height of the sequence
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def _span(start, end):
    return list(range(start, end))


# Feature indices of each hand in the 90-D input (global + head-frame groups)
LEFT_HAND_FEATURES = (
    _span(6, 12)
    + _span(24, 30)
    + _span(39, 42)
    + _span(48, 51)
    + _span(54, 60)
    + _span(66, 72)
    + _span(78, 81)
    + _span(84, 87)
)
RIGHT_HAND_FEATURES = (
    _span(12, 18)
    + _span(30, 36)
    + _span(42, 45)
    + _span(51, 54)
    + _span(60, 66)
    + _span(72, 78)
    + _span(81, 84)
    + _span(87, 90)
)
INPUT_MASKS = {
    "none": [],
    "no_left": LEFT_HAND_FEATURES,
    "no_right": RIGHT_HAND_FEATURES,
    "no_hands": LEFT_HAND_FEATURES + RIGHT_HAND_FEATURES,
}

SAMPLE_KEYS = ("input", "local_pose", "global_pose", "joints", "foot_contact")


def load_split(data_root, split, min_frames=None, subsets=None):
    """Load every ``<data_root>/<split>/*.pt`` file into a list of sequences."""
    data_root = Path(data_root)
    files = sorted((data_root / split).glob("*.pt"))
    if subsets:
        files = [f for f in files if f.stem in set(subsets)]
    if not files:
        raise FileNotFoundError(
            f"No '{split}' data found under {data_root}. Run "
            "`python -m training.prepare_data` first (see training/README.md)."
        )
    sequences = []
    for path in files:
        for seq in torch.load(path, weights_only=False):
            if min_frames is not None and seq["input"].shape[0] < min_frames:
                continue
            seq.setdefault("subset", path.stem)
            sequences.append(seq)
    return sequences


def make_sample(seq, start, end, mask_indices=(), mask_frames=None):
    """Cut ``seq[start:end]`` into a sample; optionally zero the input
    features ``mask_indices`` (over all frames, or over ``mask_frames``)."""
    sample = {key: seq[key][start:end] for key in SAMPLE_KEYS}
    inputs = sample["input"].clone()
    if len(mask_indices) > 0:
        if mask_frames is None:
            inputs[:, mask_indices] = 0
        else:
            inputs[mask_frames[0] : mask_frames[1], mask_indices] = 0
    sample["input"] = inputs
    num_frames = sample["input"].shape[0]
    sample["betas"] = seq["betas"].reshape(1, -1).repeat(num_frames, 1)
    sample["floor_height"] = seq["floor_height"].reshape(())
    return sample


class WindowDataset(Dataset):
    """Random fixed-length windows for training.

    Each item draws a random window of ``seq_len`` frames from a sequence.
    With probability ``mask_prob = (left, right, both)`` the corresponding
    hand features are zeroed over a random contiguous sub-window, which
    simulates hands leaving the headset's field of view.
    """

    def __init__(self, sequences, seq_len, repeat=1, mask_prob=(0.2, 0.2, 0.2)):
        self.sequences = [s for s in sequences if s["input"].shape[0] >= seq_len]
        if not self.sequences:
            raise ValueError(f"no sequence is at least {seq_len} frames long")
        self.seq_len = int(seq_len)
        self.repeat = int(repeat)
        self.mask_prob = tuple(mask_prob)

    def __len__(self):
        return len(self.sequences) * self.repeat

    def _random_mask(self):
        p_left, p_right, p_both = self.mask_prob
        p = torch.rand(1).item()
        if p < p_left:
            indices = LEFT_HAND_FEATURES
        elif p < p_left + p_right:
            indices = RIGHT_HAND_FEATURES
        elif p < p_left + p_right + p_both:
            indices = INPUT_MASKS["no_hands"]
        else:
            return (), None
        high = min(31, self.seq_len // 2)
        low = min(10, high - 1)
        length = int(torch.randint(low, max(low + 1, high), (1,)))
        start = int(torch.randint(0, self.seq_len - length, (1,)))
        return indices, (start, start + length)

    def __getitem__(self, idx):
        seq = self.sequences[idx % len(self.sequences)]
        num_frames = seq["input"].shape[0]
        start = 0 if num_frames <= self.seq_len else int(torch.randint(0, num_frames - self.seq_len, (1,)))
        mask_indices, mask_frames = self._random_mask()
        return make_sample(seq, start, start + self.seq_len, mask_indices, mask_frames)


class SequenceDataset(Dataset):
    """Whole sequences for evaluation (use ``batch_size=1``).

    ``input_mask`` (``none``, ``no_left``, ``no_right``, ``no_hands``) zeroes
    the corresponding hand features for the entire sequence.
    """

    def __init__(self, sequences, input_mask="none"):
        if input_mask not in INPUT_MASKS:
            raise ValueError(f"input_mask must be one of {list(INPUT_MASKS)}")
        self.sequences = list(sequences)
        self.mask_indices = INPUT_MASKS[input_mask]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        sample = make_sample(seq, 0, seq["input"].shape[0], self.mask_indices)
        sample["name"] = f"{seq.get('subset', '')}/{seq.get('source', idx)}"
        return sample
