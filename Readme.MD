# Score-Weighted Class Activation Maps (SCAM-NET)

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
