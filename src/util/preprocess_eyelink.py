import cv2
import numpy as np
import math
from skimage.transform import rotate, resize
import dlib
from imutils import face_utils
from facenet_pytorch import MTCNN
import torch
import tensorflow as tf
from tensorflow.keras.models import model_from_json


class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.

        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.

        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            box = [int(b) for b in boxes[box_ind][0]]
            x_min = max(0, box[1])
            x_max = min(box[3], frame.shape[0])
            y_min = max(box[0], 0)
            y_max = min(box[2], frame.shape[1])
            faces.append(frame[x_min:x_max, y_min:y_max])

        return faces


class EmotionPredictor(object):

    def __init__(self):

        self.EXTRACTED_WIDTH = 128
        self.EXTRACTED_HEIGHT = 160
        self.NUMBER_OF_FRAMES = 285
        self.UNIFORM = np.array(
            [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124, 126,
             127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241,
             243, 247, 248, 249, 251, 252, 253, 254, 255])

        self.landmarks_predictor = dlib.shape_predictor(cwd + "/shape_predictor_68_face_landmarks.dat")
        self.fast_mtcnn = FastMTCNN(
            stride=3,
            resize=1,
            margin=16,
            factor=0.709,
            select_largest=True,
            keep_all=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = None

    def __lbp_sip(self, frames3D):
        # frames3D: multiple grayscale videoframes (number_of_frames, height, width)
        # Returns: LBP-SIP histograms for every frame (number_of_frames, 20)
        # https://link.springer.com/content/pdf/10.1landmarks007%2F978-3-319-16865-4_34.pdf

        frames3D = np.asarray(frames3D)
        frames_n = frames3D.shape[0]
        b = np.roll(frames3D, 1, axis=0)
        m = frames3D
        f = np.roll(frames3D, -1, axis=0)

        r = np.roll(frames3D, 1, axis=2)
        l = np.roll(frames3D, -1, axis=2)
        u = np.roll(frames3D, -1, axis=1)
        d = np.roll(frames3D, 1, axis=1)

        b = (b > m).astype(int)
        f = (f > m).astype(int)
        r = (r > m).astype(int)
        l = (l > m).astype(int)
        u = (u > m).astype(int)
        d = (d > m).astype(int)

        lbp_plane_1 = (u * 2 ** 0 + l * 2 ** 1 + d * 2 ** 2 + r * 2 ** 3)
        lbp_plane_2 = (b * 2 ** 0 + f * 2 ** 1)
        lbp_plane_1 = lbp_plane_1[1:-1]
        lbp_plane_2 = lbp_plane_2[1:-1]

        histograms = np.zeros((frames_n - 2, 20))
        for i in range(frames_n - 2):
            hist_1 = self.__convert_frames_to_histogram_2(lbp_plane_1[i], 16)
            hist_2 = self.__convert_frames_to_histogram_2(lbp_plane_2[i], 4)
            histograms[i] = np.append(hist_1, hist_2, axis=0)

        return histograms

    def __convert_frames_to_histogram_2(self, lbp_plane, bins):
        # lbp_plane: LBP-Plane (height, width)
        # bins: number of bins for histogram
        # Returns histogram for LBP plane (bins,)

        idx, values = np.unique(lbp_plane, return_counts=True)
        idx, values = idx.astype(int), values.astype(int)
        hist = np.zeros(bins)
        hist[idx] = values
        hist = hist / hist.sum()

        return hist

    def __extract_frame(self, frame, cx, cy, width, height):
        # frame: Single frame (height_of_frame, width_of_frame)
        # cx, cy: midpoint of extracted frame
        # width, height: dimensions of extracted frame
        # Returns extracted frame (height, frame)
        x_min = int(cx - width / 2)
        x_max = int(cx + width / 2)
        y_min = int(cy - height / 2)
        y_max = int(cy + height / 2)

        if x_min < 0:
            x_max -= x_min
            x_min = 0
        if x_max > frame.shape[1]:
            x_min -= (x_max - frame.shape[1])
            x_max = frame.shape[1]
        if y_min < 0:
            y_max -= y_min
            y_min = 0
        if y_max > frame.shape[0]:
            y_min -= (y_max - frame.shape[0])
            y_max = frame.shape[0]

        return frame[y_min:y_max, x_min:x_max]

    def __rotate_around_point(self, points_xy, c_xy, degrees):
        # points_xy: numpy matrix of shape (n,2) with x and y coordinates for n points
        # c_xy: point to rotate around
        # degrees: degrees to rotate
        # Returns rotated points
        radians = math.radians(-degrees)
        s = math.sin(radians)
        c = math.cos(radians)
        rot_mat = np.array([[c, s], [-s, c]])
        (cx, cy) = c_xy

        points_xy[:, 0] -= cx
        points_xy[:, 1] -= cy
        points_xy = np.dot(points_xy, rot_mat)
        points_xy[:, 0] += cx
        points_xy[:, 1] += cy
        points_xy = points_xy.astype(int)

        return points_xy

    @staticmethod
    def __calculate_degrees(p1, p2):
        # p1, p2: landmark points to calculate degrees from
        # Returns degrees betweeen points
        radians = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        degrees = math.degrees(radians)

        return degrees

    def __extract_faces_from_video(self, filename):
        # filename: path
        # Returns extracted frames of faces and 68 facial landmarks for extracted frames

        # Read frames
        cap = cv2.VideoCapture(filename)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        frames = np.array(frames)

        faces = self.fast_mtcnn(frames)
        faces = np.array(faces)

        landmarks = np.zeros((faces.shape[0], 68, 2))

        for i, face in enumerate(faces):
            faceBoxRectangleS = dlib.rectangle(left=0, top=0, right=face.shape[1], bottom=face.shape[0])
            landmarks[i] = face_utils.shape_to_np(self.landmarks_predictor(face, faceBoxRectangleS))

        return faces, landmarks

    def __extract_eyes_and_mouth(self, faces, landmarks):
        # faces: multiple colored frames of faces (number_of_frames, height, width, 3)
        # landmarks: matching landmarks for faces (number_of_frames, 68, 2)
        # Returns: Extract eyes and mouth in color
        eye_width, eye_height = 60, 36  # 128, 160
        mouth_width, mouth_height = 84, 54  # 128, 160
        faces_n = faces.shape[0]

        l_eye = np.zeros((faces_n, eye_height, eye_width, 3)).astype(np.uint8)
        r_eye = np.zeros((faces_n, eye_height, eye_width, 3)).astype(np.uint8)
        mouth = np.zeros((faces_n, mouth_height, mouth_width, 3)).astype(np.uint8)

        for i in range(faces_n):
            l_eye[i] = self.__extract_frame(faces[i], landmarks[i, 37, 0], landmarks[i, 37, 1], eye_width, eye_height)
            r_eye[i] = self.__extract_frame(faces[i], landmarks[i, 44, 0], landmarks[i, 44, 1], eye_width, eye_height)
            mouth[i] = self.__extract_frame(faces[i], landmarks[i, 51, 0],
                                            min(landmarks[i, 51, 1], self.EXTRACTED_HEIGHT - int(mouth_height / 2)),
                                            mouth_width, mouth_height)

        return l_eye, r_eye, mouth

    def __rotate_and_resize(self, faces, landmarks):
        # faces: multiple color frames (number_of_frames, height, width, 3)
        # landmarks: matching landmarks for faces (number_of_frames, 68, 2)
        # return rotated and resized faces and matching landmarks
        rr_landmarks = landmarks
        rr_faces = np.zeros((faces.shape[0], self.EXTRACTED_HEIGHT, self.EXTRACTED_WIDTH, 3)).astype(np.uint8)
        a = np.array([self.EXTRACTED_WIDTH, self.EXTRACTED_HEIGHT])
        for i, face in enumerate(faces):
            degrees = self.__calculate_degrees(landmarks[i, 45], landmarks[i, 36])
            cx = rr_landmarks[i, 27, 0]
            cy = rr_landmarks[i, 27, 0]
            rr_faces[i] = cv2.resize(face, (self.EXTRACTED_WIDTH, self.EXTRACTED_HEIGHT), interpolation=cv2.INTER_AREA)
            rr_faces[i] = rotate(rr_faces[i], degrees, preserve_range=True, center=(cx, cy))
            b = np.array([face.shape[1], face.shape[0]])
            rr_landmarks[i] = rr_landmarks[i] * a / b
            rr_landmarks[i] = self.__rotate_around_point(rr_landmarks[i], (cx, cy), degrees)

        return rr_faces, rr_landmarks

    @staticmethod
    def __split_frames(faces, x_splits, y_splits):
        # faces: multiple color frames (number_of_frames, height, width, 3)
        # x_splits, y_splits: number of splits for x and y dimensions
        # Returns a list of splits
        return np.split(np.concatenate(np.split(faces, y_splits, axis=2), axis=1), x_splits * y_splits, axis=1)

    def load_model(self, model, model_weights):
        # model: path to json model
        # weights: hdf5 format weights for the model
        json_file = open(model, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(model_weights)

        self.model = model

    def main(self, filename):
        # Extract faces from video
        faces, landmarks = self.__extract_faces_from_video(filename)
        eye_distance = np.linalg.norm(landmarks[:, 39] - landmarks[:, 42], axis=1)

        # Rotate and resize
        rr_faces, rr_landmarks = self.__rotate_and_resize(faces, landmarks)

        # Extract facial parts
        l_eye, r_eye, mouth = self.__extract_eyes_and_mouth(rr_faces, rr_landmarks)

        l_eye_splits = self.__split_frames(l_eye, 3, 3)
        r_eye_splits = self.__split_frames(r_eye, 3, 3)
        mouth_splits = self.__split_frames(mouth, 3, 3)
        all_splits = l_eye_splits + r_eye_splits + mouth_splits

        # Create histograms
        histograms = self.__convert_list_to_histogram(all_splits)
        # prediction = self.__predict_video_emotion(histograms)
        return rr_faces, histograms
