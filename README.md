# CT-to-STL Vascular Modeling Tool

## Overview

This project explores automated generation of aortic geometries from image data obtained from Stanford's Vascular Model Repository. The goal was to learn medical image processing, deep learning, and geometry generation.

The program processes vascular imaging data, applies 3D image segmentation techniques, and converts predicted vascular structures into STL surface geometries for computational modeling applications.

## Workflow

1. Load VTI data
2. Preprocess image volumes and segmentation masks
3. Train a 3D U-Net segmentation model
4. Generate segmentation predictions
5. Extract surfaces using Marching Cubes
6. Smooth and refine the generated geometry
7. Export an STL file

## Libraries

* PyTorch
* VTK
* NumPy
* Scikit-learn

## Future Improvements

* Expand the size of training data
* Improve segmentation accuracy and validation
* Add configurable preprocessing parameters
* Improve mesh quality and post-processing
* Support additional anatomies
* Explore integration with SimVascular
