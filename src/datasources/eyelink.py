"""UnityEyes data source for gaze estimation."""
import os
from threading import Lock
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

from core import BaseDataSource
from models import ELG
from .eyelink_frames import EyeLinkFrames
from .frames import FramesSource
import util.gaze
import util.heatmap


class EyeLink(BaseDataSource):
    """EyeLink data loading class."""

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 batch_size: int,
                 dataset_root: str,
                 testing=False,
                 eye_image_shape=(36, 60),
                 **kwargs):

        """Create queues and threads to read and preprocess data."""
        self._short_name = 'EyeLink'
        if testing:
            self._short_name += ':test'

        # Create global index over all specified keys
        self._dataset_root = dataset_root

        self._mutex = Lock()
        self._current_index = 0

        self._frame_source = EyeLinkFrames(tensorflow_session, batch_size, dataset_root,
                                           testing, eye_image_shape, **kwargs)

        super().__init__(tensorflow_session, batch_size=batch_size,
                         testing=testing, **kwargs)

    @property
    def num_entries(self):
        """Number of entries in this data source."""
        return self._frame_source.num_entries

    @property
    def short_name(self):
        """Short name specifying source EyeLink."""
        return self._short_name

    def reset(self):
        """Reset index."""
        with self._mutex:
            super().reset()
            self._current_index = 0

    def entry_generator(self, yield_just_one=False):
        """Read entry from EyeLink dataset."""
        while True:
            for session in os.scandir(self._dataset_root):
                for sub in sorted(os.scandir(session), key=lambda x: x.path):
                    data_path = sub.path + os.sep + "log.txt"
                    if not os.path.isfile(data_path):
                        raise Exception(f"log is missing in {sub.path}")
                    with open(data_path) as f:
                        for tmp in f:
                            index, time, look_coord, click_coord, is_clicking = self._parse(tmp)
                            entry = {
                                'index': index,
                                'subsession': sub.name,
                                'look_coord': look_coord,
                                'click_coord': click_coord,
                                'is_clicking': is_clicking,
                            }
                            assert entry['full_image'] is not None
                            yield entry

    @staticmethod
    def _parse(frame):
        """ parse The eyelink data from Elías """
        [index, time, look_coord] = frame.split(maxsplit=2)
        look_coord = look_coord.split(')(')
        click_coord = (0.0, 0.0)
        is_clicking = len(look_coord) == 2
        if len(look_coord) == 1:
            look_coord = look_coord[0]
        elif len(look_coord) == 2:
            click_coord = eval('(' + look_coord[1].replace(' ', ',', 1))
            look_coord = look_coord[0] + ')'
        return eval(index), time, eval(look_coord), click_coord, is_clicking

    def old_entry_generator(self, yield_just_one=False):
        """Read entry from EyeLink dataset."""

        def get_entry(session, sub=None):
            path = f"{self._dataset_root}/{session}/"
            if sub is not None:
                path += f"{sub}/"
            video_path = path + "myndband.avi"
            data_path = path + "log.txt"
            if not os.path.isfile(video_path) or not os.path.isfile(data_path):
                raise Exception(f"myndband or log is missing in {path}")
            player = cv.VideoCapture(video_path)
            with open(data_path) as f:
                for tmp in f:
                    index, time, look_coord, click_coord, is_clicking = parse(tmp)
                    not_done, frame = player.read()
                    if not not_done:
                        raise Exception(f"The video {video_path} is shorter than its logfile suggests")
                    entry = {
                        'full_image': cv.cvtColor(frame, cv.COLOR_RGB2GRAY),
                        'look_coord': look_coord,
                        'click_coord': click_coord,
                        'is_clicking': is_clicking,
                    }
                    assert entry['full_image'] is not None
                    yield entry
            not_done, _ = player.read()
            if not_done:
                raise Exception(f"The video {video_path} is longer than its logfile suggests")
            player.release()

        try:
            while range(1) if yield_just_one else True:
                with self._mutex:
                    if self._current_index >= self.num_entries:
                        if self.testing:
                            break
                        else:
                            self._current_index = 0
                    current_index = self._current_index
                    self._current_index += 1

                for session in sorted(os.listdir(self._dataset_root)):
                    for sub in sorted(os.listdir(f"{self._dataset_root}/{session}")):
                        # special case for datapoints in shallow directories
                        if sub in {"myndband.avi", "log.txt"}:
                            sub = None
                        for entry in get_entry(session, sub):
                            yield entry
                        if sub is None:
                            continue

        finally:
            # Execute any cleanup operations as necessary
            pass
