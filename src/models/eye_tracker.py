"""ELG architecture."""
from typing import Dict

import numpy as np
import scipy
import tensorflow as tf

from core import BaseDataSource, BaseModel
from models import ELG


def _tf_mse(x, y):
    """Tensorflow call for mean-squared error."""
    return tf.reduce_mean(tf.math.squared_difference(x, y))


class EyeTracker(BaseModel):
    def __init__(self, tensorflow_session=None, first_layer_stride=1,
                 num_modules=2, num_feature_maps=32, **kwargs):
        """Specify ELG-specific parameters."""
        self._elg = ELG(
            tensorflow_session,
            first_layer_stride,
            num_feature_maps,
            num_modules,
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'heatmaps_mse': ['hourglass'],
                        'radius_mse': ['radius'],
                    },
                    'learning_rate': 1e-3,
                },
            ],
            **kwargs,
        )

        # Call parent class constructor
        super().__init__(tensorflow_session, **kwargs)

    @property
    def identifier(self):
        """Identifier for model based on data sources and parameters."""
        first_data_source = next(iter(self._train_data.values()))
        input_tensors = first_data_source.output_tensors
        if self._data_format == 'NHWC':
            _, eh, ew, _ = input_tensors['eye'].shape.as_list()
        else:
            _, _, eh, ew = input_tensors['eye'].shape.as_list()
        return 'Eye_Tracker_i%dx%d_f%dx%d_n%d_m%d' % (
            ew, eh,
            int(ew / self._hg_first_layer_stride),
            int(eh / self._hg_first_layer_stride),
            self._hg_num_feature_maps, self._hg_num_modules,
        )

    def train_loop_pre(self, current_step):
        """Run this at beginning of training loop."""
        pass

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        # initialize ELG submodel
        self._elg.initialize_if_not(training=False)
        self._elg.checkpoint.load_all()

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x1 = input_tensors['eyes']
        x2 = input_tensors['click_coord']
        x3 = input_tensors['is_clicking']
        y1 = input_tensors['look_coord'] if 'look_coord' in input_tensors else None

        with tf.compat.v1.variable_scope('input_data'):
            for i, eye in enumerate(x1):
                self.summary.feature_maps(f'eye-{i}', eye, data_format=self._data_format_longer)
            self.summary.feature_maps('click_coord', x2, data_format=self._data_format_longer)
            self.summary.feature_maps('is_clicking', x3, data_format=self._data_format_longer)
            if y1 is not None:
                self.summary.feature_maps('look_coord', y1, data_format=self._data_format_longer)

        with tf.compat.v1.variable_scope('eye_preprocessing'):
            for i, eye in enumerate(x1):
                output = self._tensorflow_session.run(
                    fetches={
                        **self._elg.output_tensors['train'],
                        **eye,
                    },
                    feed_dict={
                        self.is_training: False,
                        self.use_batch_statistics: True,
                    },
                )
        x4 = output['']

        outputs = {}
        loss_terms = {}
        metrics = {}

        print(1//0)
        # Define outputs
        return outputs, loss_terms, metrics
