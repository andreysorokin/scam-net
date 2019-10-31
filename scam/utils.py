import numpy as np
from skimage.transform import resize


def resize_activations(enhanced_model_output, input_shape):
    resized_activations = []
    for i in range(enhanced_model_output.shape[-1]):
        # resizing every activation map to original input image spatial dimensions
        resized_activations.append(resize(enhanced_model_output[..., i], input_shape, preserve_range=True))
    return np.array(resized_activations)


def normalize_activations(activation_maps):
    flattened = activation_maps.reshape((activation_maps.shape[0], -1))
    # min/max for each map
    max_a = np.max(flattened, axis=1)
    min_a = np.min(flattened, axis=1)

    # make norm = 1 where diff is zero (instead of adding/removing small epsilon)
    diffs = np.where(max_a > min_a, max_a - min_a, 1)
    return (activation_maps - min_a.reshape((-1, 1, 1))) / diffs.reshape((-1, 1, 1))
