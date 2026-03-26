# Common Configuration Files

This directory contains common configuration settings used across different models and datasets in MMDetection.

## Types of Configurations

- **lsj (Large Scale Jittering)**  
  Refers to a data augmentation technique where images are resized with large scale variations to improve robustness.

- **ms (Multi-Scale Training)**  
  Uses multiple image scales during training to improve model generalization.

- **ssj (Standard Scale Jittering)**  
  Applies moderate scale variations during training for balanced augmentation.

## Purpose

These configuration files define dataset preprocessing and augmentation strategies that are reused across different models.

## Notes

- These configs help standardize training pipelines.
- They are shared across multiple model configurations.
- Refer to specific model configs for usage examples.