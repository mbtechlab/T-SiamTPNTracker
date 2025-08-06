import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class UAV123_10fpsDataset(BaseDataset):
    """ UAV123_10fps dataset.

    Dataset structure:
        - Frames: {base_path}/data_seq/{sequence_name}/*.jpg
        - Annotations: {base_path}/anno/{sequence_name}.txt
    """
    def __init__(self, split='all'):
        super().__init__()
        self.base_path = self.env_settings.uav123_10fps_path
        self.split = split
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # Path to annotation file
        anno_path = os.path.join(self.base_path, 'anno', f'{sequence_name}.txt')
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # Path to frames
        frames_path = os.path.join(self.base_path, 'data_seq', sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))  # Sorting frames numerically
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'uav123_10fps', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # List of sequences is derived from the 'data_seq' directory
        sequences_path = os.path.join(self.base_path, 'data_seq')
        sequence_list = [seq for seq in os.listdir(sequences_path) if os.path.isdir(os.path.join(sequences_path, seq))]

        if self.split != 'all':
            # If partitioning is needed in the future, this section can be developed further
            # For example:
            # if self.split == 'train':
            #     # Select a subset of sequences for training
            # elif self.split == 'val':
            #     # Select a subset of sequences for validation
            pass  # Currently, returns all sequences

        return sequence_list
