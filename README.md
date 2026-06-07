# CT-to-STL Vascular Modeling Tool

## Overview

This project explores automated generation of aortic geometries from medical imaging datasets obtained from the Vascular Model Repository (VMR). The goal was to learn medical image processing, deep learning-based segmentation, and geometry generation.

The program processes vascular imaging data, applies 3D image segmentation techniques, and converts predicted vascular structures into STL surface geometries suitable for computational modeling applications.

## Features

* Loads vascular imaging datasets stored in VTI format
* Processes imaging data using VTK and NumPy
* Trains a 3D U-Net segmentation model using PyTorch
* Predicts vascular anatomy from medical imaging data
* Generates surface meshes using the Marching Cubes algorithm
* Exports STL geometry files for visualization and modeling

## Workflow

1. Load VTI imaging datasets
2. Preprocess image volumes and segmentation masks
3. Train a 3D U-Net segmentation model
4. Generate segmentation predictions
5. Extract vascular surfaces using Marching Cubes
6. Smooth and refine generated geometry
7. Export STL file

## Technologies Used

* Python
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
* Explore integration with SimVascular modeling workflows

## Disclaimer

This project was developed as a learning exercise and is not intended for professional use.
