"""UnityEyes data source for gaze estimation."""
import os
from threading import Lock
import time

import cv2 as cv
import numpy as np
import tensorflow as tf

from core import BaseDataSource
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

        from datasources import EyeLinkFrames
        self._eyelink_frames = EyeLinkFrames(
            tensorflow_session,
            batch_size=batch_size,
            data_format='NCHW',
            dataset_root=dataset_root,
            eye_image_shape=eye_image_shape,
        )

        # Define model
        from models import ELG
        self._elg = ELG(
            tensorflow_session,
            first_layer_stride=1,
            num_feature_maps=32,
            num_modules=2,
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    'learning_rate': 1e-3,
                },
            ],

            # Data sources for training (and testing).
            train_data={'eyelink': self._eyelink_frames},
        )

        self._inferrer = self._elg.inference_generator()

        super().__init__(tensorflow_session, batch_size=batch_size,
                         testing=testing, **kwargs)


    @property
    def num_entries(self):
        """Number of entries in this data source."""
        if self._memoized_entry_count is not None:
            return self._memoized_entry_count
        else:
            result = 0
            for session in sorted(os.listdir(self._dataset_root)):
                try:
                    # see if session length file has been generated and read it if so
                    with open(f"{self._dataset_root}/{session}/length.txt", "r") as f:
                        result += int(f.read())
                except FileNotFoundError:
                    sesslen = 0
                    for sub in sorted(os.listdir(f"{self._dataset_root}/{session}")):
                        # count and accumulate frames in each subsession
                        with open(f"{self._dataset_root}/{session}/{sub}/log.txt") as f:
                            for i, _ in enumerate(f):
                                pass
                            sesslen += i
                    result += sesslen
                    with open(f"{self._dataset_root}/{session}/length.txt", "w") as f:
                        print(sesslen, file=f)
            return result

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
        """Generate eye image entries by detecting faces and facial landmarks."""
        while range(1) if yield_just_one else True:
            for session in os.scandir(self._dataset_root):
                for sub in sorted(os.scandir(session), key=lambda x: x.path):
                    if sub.name == "length.txt":
                        continue
                    data_path = sub.path + os.sep + "log.txt"
                    if not os.path.isfile(data_path):
                        raise Exception(f"log is missing in {sub.path}")
                    with open(data_path) as f:
                        for line in f:
                            i, time, look_coord, click_coord, is_clicking = self._parse(line)
                            output = next(self._inferrer)
                            yield {
                                # TODO: þetta þarf að verða tensor eða eitthvað álíka
                                # 'video_frame_index': i,
                                # 'subsession': sub.name,
                                **output,
                                'look_coord': np.asarray(look_coord),
                                'click_coord': np.asarray(click_coord),
                                'is_clicking': np.bool_(is_clicking),
                            }

    def preprocess_entry(self, entry):
        return entry

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
