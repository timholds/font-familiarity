The readme is a project notes section for now.

**Problem Statement**: how do we get the most similar fonts to a given font?
[ ] data augmentation - vary positioning of the text
[ ] try a clip model of same font different font?
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5
[ ] do we get anything out of top eigenvectors of the data covariance matrix
[ ] distance between their mean images.

# main parts
- html files to render the text into fonts
- flask server to put the baked html onto a webpage
- training data generator that takes screenshots of those webpages
- model architecture defined with pytorch 
- file to load data and model and do the training  
- some other file to do the similarity computation

# questions
- how many datapoints per class do I want if I have around 700 classes? cifar1000 archs probably a good place to start


# TODO
[ X ] get the script reading the text from the lorum_ipsom.txt file
[ X ] get the fonts read in to the html page from fonts.txt
[ X ] get the script to render the text on an html page in the correct font
[ X ] figure out how much data I need
[ X ] get the screenshots saving with minimal overlap
[ X ] collect a dataset of 1000
[  ] add some data augmentations of different font sizes

# ML steps and ideas
[ ] train a distance model on cifar to make sure the idea works
[ ] figure out how to get the mean image of a class
[ ] how many examples per class do we need

"Data Impressions: Mining Deep Models to
Extract Samples for Data-free Applications" (2021)
- probably makes sense to use this approach and existing vision model
- use Dirichlet distribution to model prob(softmax output | class, trained model)
- dir sample space is probability over the classes
- "compute a normalized class similarity matrix (C) using the weights W connecting the final (softmax) and the pre-final layers"
- "concentration parameter (α) of the Dirichlet distribution" is the main juice here

no access to training data, "synthesizes pseudo samples from the underlying data
distribution on which it is trained."

do contrastive losses mean anything for us? could we classify the fonts into families and use that as an (additional) label?

[ ] figure out how to get the images into a nice shape for ML
- what resolution should i use 
- should i be using PIL to create the images directly and use the google fonts api to download the fonts files locally?

Update whole shebang into a data generating parameter
Create a separate system for training the network once you have the data

## how much text data do I need for 1000 images per class?
how do I know how long to make lorum_ipsum?
ensure some overlap of screenshots with scroll_height variable
- want 1000 images per class
- screenshots fixed height of 512px, scroll length 400 pixels 
    - scroll length < screenshot size is good to get some positional variation. and we can looop through the text multiple times and the positions will be in a different place 
so we need 1000*40 = 40,000 pixels worth of text
- font size of 24px with line height of 1.5 mean each line takes 36 pixels
Number of lines needed ≈ 400,000/36 ≈ 11,111 lines

I have 500 paragraphs, at an average of 10 lines each, thats 5000 lines or 5000*36 ≈ 180,000 pixels worth of text, which is about half of what we were shooting for. The code should loop back to the beginning and keep screenshotting, which is fine for now

try some different text sizes for data augmentation

# Process
Used this website to generate a bunch of text https://lipsumhub.com/?type=paragraph&length=100

Did it a few times and grabbed some different text
used this repo https://github.com/honeysilvas/google-fonts to get a full list of fonts and asked claude to put each font on it's own line, which handled the two word fonts pretty nicely