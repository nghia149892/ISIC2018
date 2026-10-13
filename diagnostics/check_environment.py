import torch
# Environment diagnostic; this file is not part of the training pipeline.
import torchvision
import numpy
import cv2
import PIL
import pandas
import sklearn

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("NumPy:", numpy.__version__)
print("OpenCV:", cv2.__version__)
print("Pillow:", PIL.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("CUDA:", torch.cuda.is_available())
