# AI-4-Media-23-24

### Setup
For Python installation instructions see: [https://git.arts.ac.uk/lmccallum/installing-python](https://git.arts.ac.uk/lmccallum/installing-python)

After making a conda environment named AIM (AI for Media) and activating it, run the command:

`conda create --name aim python=3.9`

`conda activate aim`

Then in run this (from the main folder for this repository).

`pip install -r requirements.txt`


**If you have an NVIDIA GPU**, you may want to install the CUDA version of pytorch for you machine. Uninstall pytorch (`pip uninstall torch torchvision torchaudio torchtext`) and then install the CUDA version for your machine by following the [install guide on the pytorch website homepage](https://pytorch.org/#:~:text=Aid%20to%20Ukraine.-,INSTALL%20PYTORCH,-Select%20your%20preferences). You may also need to [install CUDA](https://developer.nvidia.com/cuda-downloads). 