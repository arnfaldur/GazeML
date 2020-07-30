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


# function to parse The eyelink data from Elías
def parse(frame):
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


class EyeLink(FramesSource):
    """EyeLink data loading class."""

    def __init__(self,
                 tensorflow_session: tf.compat.v1.Session,
                 batch_size: int,
                 dataset_root: str,
                 testing=False,
                 eye_image_shape=(36, 60),
                 heatmaps_scale=1.0,
                 **kwargs):

        """Create queues and threads to read and preprocess data."""
        self._short_name = 'EyeLink'
        if testing:
            self._short_name += ':test'

        # Create global index over all specified keys
        self._dataset_root = dataset_root

        self._mutex = Lock()
        self._current_index = 0
        self._memoized_entry_count = None

        # Call parent of parent class constructor as Video.init isn't relevant
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing,
                         eye_image_shape=eye_image_shape, **kwargs)

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

    def frame_generator(self):
        """Read entry from EyeLink dataset."""

        def get_video_frames(dir):
            video_path = dir.path + os.sep + "myndband.avi"
            if not os.path.isfile(video_path):
                raise Exception(f"myndband is missing in {dir.path}")
            player = cv.VideoCapture(video_path)
            while True:
                not_done, frame = player.read()
                if not_done:
                    yield frame
                else:
                    break
            player.release()

        try:
            while True:
                for session in os.scandir(self._dataset_root):
                    for sub in sorted(os.scandir(session), key=lambda x: x.path):
                        for frame in get_video_frames(sub):
                            yield frame

        finally:
            # Execute any cleanup operations as necessary
            pass

    def frame_read_job(self):
        """Read frame from video (without skipping)."""
        generate_frame = self.frame_generator()
        while True:
            before_frame_read = time.perf_counter()
            try:
                bgr = next(generate_frame)
            except StopIteration:
                break
            if bgr is not None:
                after_frame_read = time.perf_counter()
                with self._read_mutex:
                    self._frame_read_queue.put((before_frame_read, bgr, after_frame_read))

        print(f'EyeLink dataset {self._dataset_root} closed.')
        self._open = False

    def entry_generator(self, yield_just_one=False):
        """Generate eye image entries by detecting faces and facial landmarks."""
        try:
            while range(1) if yield_just_one else True:
                # Grab frame
                with self._proc_mutex:
                    before_frame_read, bgr, after_frame_read = self._frame_read_queue.get()
                    bgr = cv.flip(bgr, flipCode=1)  # Mirror
                    current_index = self._last_frame_index + 1
                    self._last_frame_index = current_index

                    grey = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
                    frame = {
                        'frame_index': current_index,
                        'time': {
                            'before_frame_read': before_frame_read,
                            'after_frame_read': after_frame_read,
                        },
                        'bgr': bgr,
                        'grey': grey,
                    }
                    self._frames[current_index] = frame
                    self._indices.append(current_index)

                    # Keep just a few frames around
                    frames_to_keep = 120
                    if len(self._indices) > frames_to_keep:
                        for index in self._indices[:-frames_to_keep]:
                            del self._frames[index]
                        self._indices = self._indices[-frames_to_keep:]

                # Eye image segmentation pipeline
                self.detect_faces(frame)
                self.detect_landmarks(frame)
                self.calculate_smoothed_landmarks(frame)
                self.segment_eyes(frame)
                self.update_face_boxes(frame)
                frame['time']['after_preprocessing'] = time.perf_counter()

                yield frame

        finally:
            # Execute any cleanup operations as necessary
            pass

    def entry_generator(self, yield_just_one=False):
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
