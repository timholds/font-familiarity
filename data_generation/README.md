# Quickstart

# Overview

**Goal** Generate image data of some text in different fonts, with the intention of using these images to train a machine learning model downstream.

from the project root, run python data_generation/create_font_images.py


### Data generation Process
Used this website to generate a bunch of text https://lipsumhub.com/?type=paragraph&length=100

Did it a few times and grabbed some different text
used this repo https://github.com/honeysilvas/google-fonts to get a full list of fonts and asked claude to put each font on it's own line, which handled the two word fonts pretty nicely

challenge - 50GB of image data is too big to upload past Github 5GB limit
- try 100 images per class instead of 1000
- try resizing to 256x256 instead of 512
- save the images out as jpeg instead of png and so some compression

TODO experiment with different image sizes and compression levels
TODO try saving images in greyscale

# Dataset
The images themselves are currently 256x256 grayscale. The code to generate the font images should create a balanced dataset with 1000 (configurable) images per class, saved as jpg files. 

I choose jpg instead of png so we can compress the images more aggressively on disk. Basically, this is a personal project and I don't want to take up too much space on my hard drive. 

## Data structure 
- test-data
- data
    - font-images
        - abeeze
            - image0000.jpg
            - ...
            - image0100.jpg
        - ...
        - ...
        - zeyada
            - image0000.jpg
            - ...
            - image0100.jpg


control the number of images per font:
**--text_file** - the text to be rendered in each font
**--font_file** - txt file that holds a list of fonts to render, each on their own line
**--image_resolution** - 

It accomplishes this by putting text in different fonts onto a flask app and screenshotting them. 

Main files:
lorem-ipsom.md
By default, the text it will render is in lorem ipsum

TODO move all the files needed for data generation into the folder
fonts.txt
lorem.ipsom
TODO add arg for the html canvas size to create_font_images and for the number of images to generate per class

Rerun it with a different directory than fonts-images2 and use the fonts.txt instead of full-fonts-list.txt

TODO add argparsing to prep_train_test_data.py so that it grabs the right dataset