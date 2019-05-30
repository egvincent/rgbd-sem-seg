This directory contains code from charlesCXK/Depth2HHA-python and 
deeplearningaid/curfil on GitHub. See the following pages for more information:

For HHA encoding of depth images, read more here:   
https://github.com/charlesCXK/Depth2HHA-python   
* files from this: `getHHA.py`, `utils/`, `Depth2HHA-python-LICENSE` (`LICENSE` originally)
* I added the `__init__.py` files, some import statements, and a line in `utils/rgbd_util.py` (see comment)

For preprocessing of the NYU Depth v2 dataset, read more here:   
https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset   
* files from this: `preprocess_dataset.py` (`convert.py` originally), `solarized.py`, `_structure_classes.py`
* I have modified `preprocess_dataset.py` and indicated how in the comments

this preprocessing must take place in a python 2.7 conda environment (the rest of the project
runs in python 3.7), with packages:   
h5py scipy scikit-image numpy pypng joblib opencv-python[-headless]
