"""Webcam data source for gaze estimation."""
import time

import cv2 as cv

from .frames import FramesSource


class Webcam(FramesSource):
    """Webcam frame grabbing and preprocessing."""

    def __init__(self, camera_id=0, fps=60, **kwargs):
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'Webcam'

        self._capture = cv.VideoCapture(camera_id)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self._capture.set(cv.CAP_PROP_FPS, fps)

        # Call parent class constructor
        super().__init__(**kwargs)

    def frame_generator(self):
        """Read frame from webcam."""
        while True:
            ret, bgr = self._capture.read()
            if ret:
                yield bgr

    def frame_read_job(self):
        """Read frame from webcam."""
        generate_frame = self.frame_generator()
        while True:
            before_frame_read = time.perf_counter()
            bgr = next(generate_frame)
            if bgr is not None:
                after_frame_read = time.perf_counter()
                with self._read_mutex:
                    self._frame_read_queue.queue.clear()
                    self._frame_read_queue.put_nowait((before_frame_read, bgr, after_frame_read))
