"""UnityEyes data source for gaze estimation."""
import os
from threading import Lock

import cv2 as cv
import numpy as np
import tensorflow as tf
import ujson

from core import BaseDataSource
from datasources import FramesSource
import util.gaze
import util.heatmap


# function to parse The eyelink data from Elías
def parse(frame):
    [index,time,look_coord] = frame.split(maxsplit=2)
    look_coord = look_coord.split(')(')
    click_coord = (0.0,0.0)
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

        super().__init__(tensorflow_session, batch_size, eye_image_shape=eye_image_shape)
        """Create queues and threads to read and preprocess data."""
        self._short_name = 'EyeLink'
        if testing:
            self._short_name += ':test'

        # Cache some parameters
        self._eye_image_shape = eye_image_shape
        self._heatmaps_scale = heatmaps_scale

        # Create global index over all specified keys
        self._dataset_root = dataset_root
        self._session_names = sorted(os.listdir(dataset_root))

        self._mutex = Lock()
        self._current_index = 0
        self._memoized_entry_count = None

        # Call parent class constructor
        super().__init__(tensorflow_session, batch_size=batch_size, testing=testing, **kwargs)

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
                        # special case for datapoints in shallow directories
                        if sub in {"myndband.avi", "log.txt"}:
                            with open(f"{self._dataset_root}/{session}/log.txt") as f:
                                for i, _ in enumerate(f):
                                    pass
                                sesslen += i
                            break
                        else:
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

    def preprocess_entry(self, entry):
        """Use annotations to segment eyes and calculate gaze direction."""
        full_image = entry['full_image']
        json_data = entry['json_data']
        del entry['full_image']
        del entry['json_data']

        ih, iw = full_image.shape
        iw_2, ih_2 = 0.5 * iw, 0.5 * ih
        oh, ow = self._eye_image_shape

        def process_coords(coords_list):
            coords = [eval(l) for l in coords_list]
            return np.array([(x, ih-y, z) for (x, y, z) in coords])
        interior_landmarks = process_coords(json_data['interior_margin_2d'])
        caruncle_landmarks = process_coords(json_data['caruncle_2d'])
        iris_landmarks = process_coords(json_data['iris_2d'])

        random_multipliers = []

        def value_from_type(augmentation_type):
            # Scale to be in range
            easy_value, hard_value = self._augmentation_ranges[augmentation_type]
            value = (hard_value - easy_value) * self._difficulty + easy_value
            value = (np.clip(value, easy_value, hard_value)
                     if easy_value < hard_value
                     else np.clip(value, hard_value, easy_value))
            return value

        def noisy_value_from_type(augmentation_type):
            # Get normal distributed random value
            if len(random_multipliers) == 0:
                random_multipliers.extend(
                    list(np.random.normal(size=(len(self._augmentation_ranges),))))
            return random_multipliers.pop() * value_from_type(augmentation_type)

        # Only select almost frontal images
        h_pitch, h_yaw, _ = eval(json_data['head_pose'])
        if h_pitch > 180.0:  # Need to correct pitch
            h_pitch -= 360.0
        h_yaw -= 180.0  # Need to correct yaw
        if abs(h_pitch) > 25 or abs(h_yaw) > 25:
            return None

        # Prepare to segment eye image
        left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
        right_corner = interior_landmarks[8, :2]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                              np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

        # Centre axes to eyeball centre
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-iw_2], [-ih_2]]

        # Rotate eye image if requested
        rotate_mat = np.asmatrix(np.eye(3))
        rotation_noise = noisy_value_from_type('rotation')
        if rotation_noise > 0:
            rotate_angle = np.radians(rotation_noise)
            cos_rotate = np.cos(rotate_angle)
            sin_rotate = np.sin(rotate_angle)
            rotate_mat[0, 0] = cos_rotate
            rotate_mat[0, 1] = -sin_rotate
            rotate_mat[1, 0] = sin_rotate
            rotate_mat[1, 1] = cos_rotate

        # Scale image to fit output dimensions (with a little bit of noise)
        scale_mat = np.asmatrix(np.eye(3))
        scale = 1. + noisy_value_from_type('scale')
        scale_inv = 1. / scale
        np.fill_diagonal(scale_mat, ow / eye_width * scale)
        original_eyeball_radius = 71.7593
        eyeball_radius = original_eyeball_radius * scale_mat[0, 0]  # See: https://goo.gl/ZnXgDE
        entry['radius'] = np.float32(eyeball_radius)

        # Re-centre eye image such that eye fits (based on determined `eye_middle`)
        recentre_mat = np.asmatrix(np.eye(3))
        recentre_mat[0, 2] = iw/2 - eye_middle[0] + 0.5 * eye_width * scale_inv
        recentre_mat[1, 2] = ih/2 - eye_middle[1] + 0.5 * oh / ow * eye_width * scale_inv
        recentre_mat[0, 2] += noisy_value_from_type('translation')  # x
        recentre_mat[1, 2] += noisy_value_from_type('translation')  # y

        # Apply transforms
        transform_mat = recentre_mat * scale_mat * rotate_mat * translate_mat
        eye = cv.warpAffine(full_image, transform_mat[:2, :3], (ow, oh))

        # Convert look vector to gaze direction in polar angles
        look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3]
        look_vec[0] = -look_vec[0]
        original_gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        look_vec = rotate_mat * look_vec.reshape(3, 1)
        gaze = util.gaze.vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        if gaze[1] > 0.0:
            gaze[1] = np.pi - gaze[1]
        elif gaze[1] < 0.0:
            gaze[1] = -(np.pi + gaze[1])
        entry['gaze'] = gaze.astype(np.float32)

        # Draw line randomly
        num_line_noise = int(np.round(noisy_value_from_type('num_line')))
        if num_line_noise > 0:
            line_rand_nums = np.random.rand(5 * num_line_noise)
            for i in range(num_line_noise):
                j = 5 * i
                lx0, ly0 = int(ow * line_rand_nums[j]), oh
                lx1, ly1 = ow, int(oh * line_rand_nums[j + 1])
                direction = line_rand_nums[j + 2]
                if direction < 0.25:
                    lx1 = ly0 = 0
                elif direction < 0.5:
                    lx1 = 0
                elif direction < 0.75:
                    ly0 = 0
                line_colour = int(255 * line_rand_nums[j + 3])
                eye = cv.line(eye, (lx0, ly0), (lx1, ly1),
                              color=(line_colour, line_colour, line_colour),
                              thickness=max(1, int(6*line_rand_nums[j + 4])),
                              lineType=cv.LINE_AA)

        # Rescale image if required
        rescale_max = value_from_type('rescale')
        if rescale_max < 1.0:
            rescale_noise = np.random.uniform(low=rescale_max, high=1.0)
            interpolation = cv.INTER_CUBIC
            eye = cv.resize(eye, dsize=(0, 0), fx=rescale_noise, fy=rescale_noise,
                            interpolation=interpolation)
            eye = cv.equalizeHist(eye)
            eye = cv.resize(eye, dsize=(ow, oh), interpolation=interpolation)

        # Add rgb noise to eye image
        intensity_noise = int(value_from_type('intensity'))
        if intensity_noise > 0:
            eye = eye.astype(np.int16)
            eye += np.random.randint(low=-intensity_noise, high=intensity_noise,
                                     size=eye.shape, dtype=np.int16)
            cv.normalize(eye, eye, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            eye = eye.astype(np.uint8)

        # Add blur to eye image
        blur_noise = noisy_value_from_type('blur')
        if blur_noise > 0:
            eye = cv.GaussianBlur(eye, (7, 7), 0.5 + np.abs(blur_noise))

        # Histogram equalization and preprocessing for NN
        eye = cv.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if self.data_format == 'NHWC' else 0)
        entry['eye'] = eye

        # Select and transform landmark coordinates
        iris_centre = np.asarray([
            iw_2 + original_eyeball_radius * -np.cos(original_gaze[0]) * np.sin(original_gaze[1]),
            ih_2 + original_eyeball_radius * -np.sin(original_gaze[0]),
            ])
        landmarks = np.concatenate([interior_landmarks[::2, :2],  # 8
                                    iris_landmarks[::4, :2],  # 8
                                    iris_centre.reshape((1, 2)),
                                    [[iw_2, ih_2]],  # Eyeball centre
                                    ])  # 18 in total
        landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant',
                                       constant_values=1))
        landmarks = np.asarray(landmarks * transform_mat.T)
        landmarks = landmarks[:, :2]  # We only need x, y
        entry['landmarks'] = landmarks.astype(np.float32)

        # Generate heatmaps if necessary
        if self._generate_heatmaps:
            # Should be half-scale (compared to eye image)
            entry['heatmaps'] = np.asarray([
                util.heatmap.gaussian_2d(
                    shape=(self._heatmaps_scale*oh, self._heatmaps_scale*ow),
                    centre=self._heatmaps_scale*landmark,
                    sigma=value_from_type('heatmap_sigma'),
                )
                for landmark in entry['landmarks']
            ]).astype(np.float32)
            if self.data_format == 'NHWC':
                entry['heatmaps'] = np.transpose(entry['heatmaps'], (1, 2, 0))

        return entry
