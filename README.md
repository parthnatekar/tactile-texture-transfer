# tactile-texture-transfer
Transferring real and tactile textures made by users onto digital images using neural style transfer.

## Requirements
> Tensorflow

> Keras

> OpenCV

> Numpy

> Matplotlib

## Usage

1. Clone Repository ```git clone https://github.com/parthnatekar/tactile-texture-transfer.git``` and navigate to ```TTT/```

2. Install the IP Webcam app from the Google Play Store on your mobile device.

3. Start a webcam server on the application.

4. Run ```python3 ttt.py -p PATH_TO_INPUT_IMAGE -i IP_ADDRESS_OF_MOBILE_SERVER:PORT```

5. Use flag ```-m``` (optional) for static (i.e the style image is captured only once before the transfer algorithm is run) or dynamic (the style image is captured every epoch while running the algorithm, allowing modification of target style during transfer).
Eg: ```python3 ttt.py -p PATH_TO_INPUT_IMAGE -i IP_ADDRESS_OF_MOBILE_SERVER:PORT -m d```

## Results

The image below shows an example implementation, where a user creates a texture using grains, and transfers this to an input image.

<p align="center">
  <img src="./images/Picture1.png" width="600"> 
</p>

