# NNCF compression

## General Description

This page provides information on integration of NNCF framework into OTEDetection code.

First of all, it should be noted that NNCF is not a part of OTEDetection and is not
installed by OTEDetection, and integration of OTEDetection with NNCF framework is made in a
transparent way:  
if NNCF options are not set in the config file, the NNCF framework is not used and the
OTEDetection will work "in a normal way" if NNCF is not installed.

If the config file of a model contains section `nncf_config`, and the section is not empty,
but the NNCF network is not installed, a corresponding exception will be raised.

If the config file of a model contains section `nncf_config`, the section is not empty,
and the NNCF network is installed, the section will be passed to the NNCF framework as its
config without changes.

## Typical pipeline of NNCF compression with OTEDetection

Typically, the pipeline of compression with NNCF looks as follows:

* Either a model is trained without compression or a pre-trained model is downloaded
* The model is fine-tuned with compression (the parameter `nncf_config` is set),  
  the result of this step is a snapshot with the compressed model
* (optional) The model is fine-tuned with compression a bit more
  (with the same compression parameters `nncf_config` in the config file),  
  the result of this step is a snapshot with fine-tuned compressed model
* The compressed model is tested
  (with the same compression parameters `nncf_config` in the config file)
* The compressed model is exported to ONNX/OpenVINO™
  (with the same compression parameters `nncf_config` in the config file),  
  the result of this step is a snapshot with fine-tuned compressed model

To load an uncompressed model for compression you can use, as usual, the config parameters
`load_from` and `resume_from` to pass a pre-trained checkpoint; since the checkpoint was not
trained with compression, NNCF will initialize the compression inner structures for the model
and will start training with compression.

To load a compressed model's checkpoint for fine-tuning you can use the same config parameters
`load_from` and `resume_from`: if the snapshot is received as a result of training with
compression, the NNCF will not re-initialize the compression inner structures and the fine
tuning will be made.

Note that after NNCF compression is applied to a model, the model's checkpoints should be
loaded with a config file with the same NNCF parameters: the value of the dict `nncf_config` in
the config file should not be changed at all or should be changed carefully, since after some
changes the compressed model won't be loaded.

And note that when you make testing and/or exporting to ONNX/OpenVINO™, the corresponding
scripts (`tools/test.py` and `tools/export.py`) receive the path to the model's checkpoint as a
command line parameter, the parameter overrides the value of the config parameters `load_from`
and `resume_from`.

## Example of config files for models compressed by NNCF

TBD

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
`mmdet/integration/nncf/`.
