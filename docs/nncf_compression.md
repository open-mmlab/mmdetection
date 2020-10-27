# NNCF compression

## General description

OTEDetection allows to make compression of the models by
[NNCF (Neural Network Compression Framework)](https://github.com/openvinotoolkit/nncf_pytorch).

The work of OTEDetection is checked with [**NNCF 1.4.1**](https://github.com/openvinotoolkit/nncf_pytorch/tree/v1.4.1).
Please, use this version of NNCF in case of any issues.

NNCF allows to make compression using the following methods
(including combination of some of the methods):
* int8 quantization
* int4 quantization
* binarization
* sparsity
* filter pruning

To make compression of a model, NNCF gets a pre-trained model and wraps the whole PyTorch model
and PyTorch classes used by the model (e.g. Conv2d) by its own classes.
After that to make compression a training (fine-tuning) of the model should be started --
typically, it should be done by the same code as the original model was trained.
During the fine-tuning the wrapped classes make additional operations during each training
step (e.g. in case of int8 quantization each result of convolution layers will be
quantized, for filter pruning special technique will be applied to reduce number of filters in
each convolution, etc).

The result of such fine-tuning is a compressed model that may be exported to OpenVINO™.

## Integration of NNCF into OTEDetection

Please, note that OTEDetection does not require NNCF framework to be installed for
usual training (without compression).
Integration of OTEDetection with NNCF framework is made in a transparent way:

* If NNCF parameters are not set in the config file, the NNCF framework is not used and the
  OTEDetection will work "in a normal way", no matter if NNCF is installed or is not.

* If the config file of a model contains a parameter `nncf_config`, and the parameter is a non-empty dict,
  NNCF will be used for the model compression:  
  - If NNCF **is not** installed, a corresponding exception will be raised.  
  - If NNCF **is** installed, the dict that is the value of the parameter `nncf_config`
    will be passed to the NNCF framework as its config without changes.

Example of NNCF parameter `nncf_config` that may be used for int8 quantization of `ssd300_coco`:
```python
nncf_config = {
    "input_info": {
        "sample_size": [1, 3, 1000, 600]
    },
    "compression": {
        "algorithm": "quantization",
        "initializer": {
            "range": {
                "num_init_steps": 10
            },
            "batchnorm_adaptation": {
                "num_bn_adaptation_steps": 30,
            }

        }
    },
    "log_dir": work_dir
}
```

See details on NNCF config parameters in
[documentation on NNCF config files](https://github.com/openvinotoolkit/nncf_pytorch/blob/develop/docs/ConfigFile.md).

Also you can see parameters of different NNCF compression algorithms in the
[documentation on NNCF algorithms](https://github.com/openvinotoolkit/nncf_pytorch/blob/develop/docs/Algorithms.md).

## Example of config files for models compressed by NNCF

You can see the examples of the config files in the folder [./configs/nncf_compression](./configs/nncf_compression).

## How to start/resume NNCF compression.

Typically, the pipeline of compression with NNCF looks as follows:

1. Get a usual (i.e. uncompressed) model.  
   To receive the model it may be trained without compression or a pre-trained model may be downloaded.

2. The model is fine-tuned with compression (the parameter `nncf_config` is set).  
   The result of this step is a checkpoint with the compressed model

3. (optional) The compression precess may be resumed: model may be fine-tuned with compression a bit more
   (with the same compression parameters `nncf_config` in the config file).  
   The result of this step is a checkpoint with fine-tuned compressed model

4. The compressed model may be tested
   (with the same compression parameters `nncf_config` in the config file).  
   The result of this step is quality metrics of the compressed model

5. The compressed model may be exported to ONNX/OpenVINO™
   (with the same compression parameters `nncf_config` in the config file).  
   The result of this step is a ONNX/OpenVINO™ compressed model

6. The model exported to ONNX/OpenVINO™ may be tested.  
   The result of this step is quality metrics of the exported model

To load an uncompressed model for compression you can use (as usual) the config parameter
`load_from` to pass a pre-trained checkpoint; since the checkpoint was not
trained with compression, NNCF will initialize the compression inner structures for the model
and will start training with compression.

To load a compressed model's checkpoint to resume compression you can use the same config
parameter `load_from` and the config parameter `resume_from`: if the checkpoint is received as
a result of training with compression, the NNCF will not re-initialize the compression inner
structures and the fine tuning will be made.

Note that after NNCF compression is applied to a model, the model's checkpoints should be
loaded with a config file with the same NNCF parameters: the value of the dict `nncf_config` in
the config file should not be changed at all or should be changed carefully, since after some
changes the compressed model won't be loaded.

## Additional config parameters for NNCF

At the moment also configuration parameter `nncf_should_compress_postprocessing` may be set.

This parameter is used to choose if we should try to make NNCF compression
for a whole model graph including postprocessing (`nncf_should_compress_postprocessing=True`),
or make NNCF compression of the part of the model without postprocessing
(`nncf_should_compress_postprocessing=False`).

Our primary goal is to make NNCF compression of such big part of the model as
possible, so `nncf_should_compress_postprocessing=True` is our primary choice, whereas
`nncf_should_compress_postprocessing=False` is our fallback decision.

When we manage to enable NNCF compression for sufficiently many models,
we will keep one choice only.

## Code of integration NNCF into OTEDetection

The code connecting NNCF framework with OTEDetection is placed in the folder
[mmdet/integration/nncf/](../mmdet/integration/nncf/).
