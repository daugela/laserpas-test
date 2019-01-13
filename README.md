## Laserpas test

A showcase of batch image processing from 2 separate (RGB/NIR) cameras with dirrecent zooms.

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

Install required opencv lib
`pip install opencv-python`

Pass prepared csv file (with image paths) to process script for batch processing
`python process.py batch_photo_paths.csv`