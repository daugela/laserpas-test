## Laserpas test

A showcase of batch image processing from 2 separate (RGB/NIR) cameras with different zooms using [OpenCV](https://opencv.org)

The list of input images (relative paths) should be exported into simple csv file (batch_photo_paths.csv) before execution.
Resulting images (combined overlaps between 2 inputs) will be saved into same directory.
Resulting image size will match the smaller NIR input, contain all color channels from initial images and will have a *combined_NIR_filename.png* file naming.

## Setup on a mac or some other unix based machine

Clone this repo

`git clone https://github.com/daugela/laserpas-test.git`

Go inside

`cd laserpas-test`

Be sure you have properly working python version

`which python3`

Create virtual env for package indepencency

`python3 -m venv env`

Activate environment

`source ./env/bin/activate`

Install required [OpenCV](https://pypi.org/project/opencv-python/) lib v.4

`pip install opencv-python`

Pass prepared csv file (with image paths) to process script for batch processing

`python process.py batch_photo_paths.csv`

Enjoy processed results in FinalOutput directory