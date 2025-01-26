# Quickstart
**Create the font images:**  
`python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/fonts_test.txt --output_dir test-data/font-images --samples_per_class 10 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  

**Preprocess the data:**  
`python data_generation/prep_train_test_data.py --input_image_dir test-data/font-images --output_dir test-data/font-dataset-npz --test_size .1`  

# Overview
**Goal** Generate image data of some text in different fonts, with the intention of using these images to train a machine learning model downstream.  

The code to generate the font images should create a balanced dataset with 1000 (configurable) images per class. The images are saved in a directory structure that matches the font name. The images are then preprocessed into a train/test split and saved as npz files.  

The images themselves are currently 256x256 grayscale jpg files.

### Data generation Process
I used Claude and this website to generate a bunch of text https://lipsumhub.com/?type=paragraph&length=100 and put it all into `lorem_ipsum.txt`

I used this repo https://github.com/honeysilvas/google-fonts to get a full list of fonts and asked Claude to put each font on it's own line. There are some two-word fonts that would have required some manual parsing, and Claude handled these easily. The fonts are each on their own line in `full_fonts_list.txt`

Then I render the text in different fonts onto an html page using flask app. A selenium scraper goes through the pages and screenshots the text, saving the image to a directory that matches the name of the font. 



# Main Scripts
### Create Images 
`--text_file`: the text to be rendered in each font. The default is just some lorem ipsum text and claude text but you can totally make your own.   
`font_file`: txt file that holds a list of fonts to render, each on their own line  
`--output_dir`: the directory to save the images to  
`--samples_per_font`: the number of images we generate for each font. With the full font list at 700 fonts, a good number to start with is probably between 100-1000   
`--image_resolution`: size of the images to create  
`--port`: the port to run the flask app on  
`--font_size`: the font size to render the text in  
`--line_height`: the line height to render the text in  

### Preprocess Images
`--input_image_dir`: the directory where the images are saved  
`--output_dir`: the directory to save the preprocessed images to  
`--test_size`: the proportion of the dataset to use as the test set  




## Dataset structure
`create_font_images.py` should create a directory called `test-data/font-images` with the following structure:
```
- test-data
    - font-images
        - abeeze
            - image0000.jpg
            - ...
            - image0010.jpg
        - ...
        - archivo narrow
            - image0000.jpg
            - ...
            - image0010.jpg
```

`prep_train_test_data.py` should create a directory called test-data/font-dataset-npz with the following structure
```
- test-data
    - font-images-npz
        - label_mapping.npy 
        - train.npz 
        - test.npz 
```



# Choices and tradeoffs
## Resolution Compression Experiment
### Image parameters vs filesize 
|Filesize  | Compression    | Resolution     | Container | Quality      |
| -------  | -------------- | -------------- | --------- | -------      |
~1 kb      | compression 10 | image size 128 |    128    |  too zoomed  |
2-3kb      | compression 10 | image size 256 |    128    |  too zoomed  |
| | | | |
1-2kb      | compression 10 | image size 128 |    256    |  too blurry  |
3-4kb      | compression 50 | image size 128 |    256    |    blurry    |
| | | | |
4-5kb      | compression 10 | image size 256 |    256    |blur, passable|
7-8kb      | compression 20 | image size 256 |    256    |    decent    |
11-13kb    | compression 50 | image size 256 |    256    |     good     |
| | | | |
12-13kb    | compression 10 | image size 512 |    256    |              |
15-16kb    | compression 20 | image size 512 |    256    |              |
| | | | |
4-5kb      | compression 10 | image size 256 |    512    |  too wide    |
18-20kb    | compression 10 | image size 512 |    512    |  too wide    |


**Idea** is to increase the font size instead of increasing the resolution / amount of text in each image

**Note**: Some blurring can actually be desirable for the ML task as it has a regularizing effect. In other words, blurring helps bias the parameters of the network to be closer to 0, which will lead to smoother loss landscapes and better generalization. in other words, small weights and biases mean less expressability but smoother interpolation between training points. This all reduces overfitting. 


I choose jpg instead of png so we can compress the images more aggressively on disk. Basically, this is a personal project and I don't want to take up too much space on my hard drive.  


# Challenges
challenge - 50GB of image data is too big to upload past Github 5GB limit
- try 100 images per class instead of 1000
- try resizing to 256x256 instead of 512
- save the images out as jpeg instead of png and so some compression

## how much text data do I need for 1000 images per class?
how do I know how long to make lorum_ipsum?
ensure some overlap of screenshots with scroll_height variable
- want 1000 images per class
- screenshots fixed height of 512px, scroll length 400 pixels 
    - scroll length < screenshot size is good to get some positional variation. and we can looop through the text multiple times and the positions will be in a different place 
so we need 1000*40 = 40,000 pixels worth of text
- font size of 24px with line height of 1.5 mean each line takes 36 pixels
Number of lines needed ≈ 400,000/36 ≈ 11,111 lines

I have 500 paragraphs, at an average of 10 lines each, thats 5000 lines or 5000*36 ≈ 180,000 pixels worth of text, which is about half of what we were shooting for.

