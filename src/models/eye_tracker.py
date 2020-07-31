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
    """ELG architecture as introduced in [Park et al. ETRA'18]."""

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
        return 'ELG_i%dx%d_f%dx%d_n%d_m%d' % (
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
        x = input_tensors['eye']
        y1 = input_tensors['heatmaps'] if 'heatmaps' in input_tensors else None
        y2 = input_tensors['landmarks'] if 'landmarks' in input_tensors else None
        y3 = input_tensors['radius'] if 'radius' in input_tensors else None

        with tf.compat.v1.variable_scope('input_data'):
            self.summary.feature_maps('eyes', x, data_format=self._data_format_longer)
            if y1 is not None:
                self.summary.feature_maps('hmaps_true', y1, data_format=self._data_format_longer)

        outputs = {}
        loss_terms = {}
        metrics = {}


        # Define outputs
        return outputs, loss_terms, metrics
