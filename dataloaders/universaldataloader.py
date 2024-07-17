import os.path as osp
import numpy as np
from PIL import Image
from .base import Sequence, SequenceClip_simple
from .universaldataset import UniversalDataset
from easydict import EasyDict as edict
from args import get_parser

parser = get_parser()
args = parser.parse_args()

if 'youtube' in args.dataset:
    from configs.youtube.config import db_read_sequences
else:
    from configs.davis.config import db_read_sequences


class UniversalLoader(UniversalDataset):

    def __init__(self, args, split='train', augment=False, inputRes=None, transform=None):
        self._year = args.year
        self._phase = split
        self.split = split
        self.dataset = args.dataset
        self._length_clip = args.length_clip
        self.augment = augment
        self.inputRes = inputRes
        self.transform = transform

        # Load Image Sequences
        self._db_sequences = db_read_sequences(self._year, self._phase)
        if "youtube" in self.dataset:
            self.sequences = [Sequence(self._phase, s["video_name"], regex='*.jpg') for s in self._db_sequences]
        if "davis" in self.dataset:
            self.sequences = [Sequence(self._phase, s.name, regex='*.png') for s in self._db_sequences]
        # print(len(self.sequences))
        # Load sequence clips
        self.sequence_clips = []
        self._db_sequences = db_read_sequences(self._year, self._phase)
        k = int(self._length_clip / 2)
        for seq, _ in zip(self.sequences, self._db_sequences):
            images = seq.files
            num_clips = seq._numframes
            for idx in range(num_clips):
                clip_id_list = [max(0, min(num_clips - 1, i)) for i in range(idx - k, idx + k + 1)]
                starting_frame = int((osp.splitext(osp.basename(images[clip_id_list[0]]))[0]).replace("frame_", ""))
                self.sequence_clips.append(SequenceClip_simple(seq, starting_frame, clip_id_list))

        self._keys = dict(zip([s for s in self.sequences],
                              range(len(self.sequences))))

        self._keys_clips = dict(zip([s.name + str(s.starting_frame) for s in self.sequence_clips],
                                    range(len(self.sequence_clips))))

        try:
            self.color_palette = np.array(Image.open(
                self.annotations[0].files[0]).getpalette()).reshape(-1, 3)
        except Exception as e:
            self.color_palette = np.array([[0, 255, 0]])

    def get_raw_sample_clip(self, key):
        """ Get sequences and annotations pairs."""
        if isinstance(key, str):
            sid = self._keys_clips[key]
        elif isinstance(key, int):
            sid = key
        else:
            raise ValueError("Please input the correct Key!")

        return edict({
            'images': self.sequence_clips[sid]
        })

    def sequence_name_to_id(self, name):
        """ Map sequence name to index."""
        return self._keys[name]

    def sequence_name_to_id_clip(self, name):
        """ Map sequence name to index."""
        return self._keys_clips[name]

    def sequence_id_to_name(self, sid):
        """ Map index to sequence name."""
        return self._db_sequences[sid].name

    def sequence_id_to_name_clip(self, sid):
        """ Map index to sequence name."""
        return self.sequence_clips[sid]

    def iternames(self):
        """ Iterator over sequence names."""
        for s in self._db_sequences:
            yield s

    def iternames_clips(self):
        """ Iterator over sequence names."""
        for s in self.sequence_clips:
            yield s

    def iteritems(self):
        return self.__iter__()
