from keras import Model
import numpy as np

from scam.exceptions import InvalidState
from scam.utils import resize_activations, normalize_activations


class ScoreCAM:
    def __init__(self, model_input, last_conv_output, softmax_output, input_shape, cam_batch_size=-1):
        self.model_input = model_input
        self.last_conv_output = last_conv_output
        self.softmax_output = softmax_output
        self.last_conv_model = Model(inputs=model_input, outputs=last_conv_output)
        self.softmax_model = Model(inputs=model_input, outputs=softmax_output)
        self.input_shape = input_shape
        self.cam_batch_size = cam_batch_size

        self.normalized_maps = None
        self.classes_activation_scale = None

    def prepare_cam(self, input):
        output_conv = self.last_conv_model.predict(input)
        # Only first image from convolutions will be used
        resized = resize_activations(output_conv[0], self.input_shape)
        # filter_size x input_shape[0] x input_shape[1] - resized to original input dimensions
        normalized_maps = normalize_activations(resized)

        # todo add batch support
        # repeat input
        repeat_input = np.tile(input, (normalized_maps.shape[0], 1, 1, 1))
        expanded_activation_maps = np.expand_dims(normalized_maps, axis=3)
        masked_images = np.multiply(repeat_input, expanded_activation_maps)
        # input: filter_size x input_shape[0] x input_shape[1] -> Output filter_size x Classes_Count
        self.classes_activation_scale = self.softmax_model.predict(masked_images)
        self.normalized_maps = normalized_maps

    def get_class_heatmap(self, class_id):
        if self.normalized_maps is None or self.classes_activation_scale is None:
            raise InvalidState('Call prepare_cam before accessing get_class_heatmap, '
                               'activations must be prepared via prepare_cam')
        final_weights = self.classes_activation_scale[:, class_id]
        final_maps = np.multiply(self.normalized_maps, final_weights.reshape((-1, 1, 1)))
        # ReLU
        final_maps_max = np.max(final_maps, axis=0)
        final_class_activation_map = np.where(final_maps_max > 0, final_maps_max, 0)
        return final_class_activation_map