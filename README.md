# Score-Weighted Class Activation Mapping (SCAM-NET)

## About
This is an implementation of: <cite data-cite="wang2019scorecamimproved">Score-CAM Improved Visual Explanations Via Score-Weighted Class Activation Mapping</cite>.
[https://arxiv.org/abs/1910.01279]

BibTex reference:

    @misc{wang2019scorecamimproved,
    title={Score-CAM:Improved Visual Explanations Via Score-Weighted Class Activation Mapping},
    author={Haofan Wang and Mengnan Du and Fan Yang and Zijian Zhang},
    year={2019},
    eprint={1910.01279},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
	}

It is capable of postprocessing CNNs by taking its final output convolutional layer and softmax layer and generating spatial heatmap for the specified class.
Regions with higher score correspond to the pixels with greater importance in classifying an image with a specific class.


## Usage

To use ScoreCAM class with Keras is as easy as adding 2 calls:

    from scam.keras import ScoreCAM
    scoreCAM = ScoreCAM(model_input=model_input, last_conv_output=conv_layers, softmax_output=softmax_output, input_shape=input_shape)
    scoreCAM.prepare_cam(img)

* `model_input` - is an input layer
* `conv_layers` - last convolutional layer output 
* `softmax_output` - final classification layer output.
* `input_shape` - expected image spatial dimensions (e.g. (224,224))


and
    
    # return heatmap of the same size as image
    heatmap = scoreCAM.get_class_heatmap(class_id)

## Expected Output

The output is a heatmap which describes an importance of a class `class_id` with respect to pixel location.
Below is the sample output for tiger_cat class:

![cat_dog_3_heatmap](images/cat_dog_3_out.png?raw=true "cat_dog_3_heatmap")