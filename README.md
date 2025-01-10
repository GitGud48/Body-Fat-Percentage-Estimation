
<h1> Body Fat Percentage Estimation from Images </h1>

This repository contains a machine learning model capable of estimating a person's body fat percentage based on a single image input with 75% accuracy on testing data. The model uses deep learning techniques to analyze body composition and provide an approximate body fat percentage, between 5% to 19%. 

<h2> Installation</h2>

1. Clone the repository into your local machine
```bash
 git clone https://github.com/GitGud48/Body-Fat-Percentage-Estimation
 cd Body-Fat-Percentage-Estimation
```
2. Install dependencies

Pytorch and Torchvision
(https://pytorch.org/ for more details)

Pathlib
```bash
 pip install pathlib
```
<h2> How to Use</h2>

1. Preparing Image

Make sure that the image contains only the chest to midsection, under good lighting conditions and unobstructed by arms or phones in the frame. The model will perform best when the camera is directly in front of the person. 

2. Execute

Run the file Predictions.py. This will create an empty directory called images in the same folder as this repository. Place as images as desired into the empty directory. Then enter the name of the image that the model needs to analyze. This should output the predictions made by the model. 


