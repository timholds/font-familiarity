# Quickstart
Create a new virtual environment and install the requirements. I'm using Python 3.10.12, use other versions and YMMV
```
python -m venv font-env
source font-env/bin/activate
pip install -r requirements.txt
```


To run any of the individual components from the root of font-familiarity:
- `python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/fonts_test.txt --output_dir test-data/font-images --samples_per_class 10 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  
- `python data_generation/prep_train_test_data.py --input_image_dir test-data/font-images --output_dir test-data/font-dataset-npz --test_size .1`  
- `python ml/train.py --data_dir "test-data/font-dataset-npz" --epochs 30 --batch_size 64 --learning_rate .001 --weight_decay .01 --embedding_dim 256 --resolution 64 --initial_channels 16`  
- `python create_embeddings.py --model_path test-data/font-dataset-npz/fontCNN_BS64-ED128-IC16.pt --data_dir test-data/font-dataset-npz --embeddings_file class_embeddings_512.npy`  
- `python frontend_app.py --model_path fontCNN_BS64-ED128-IC16.pt --data_dir test-data/font-dataset-npz --embeddings_path test-data/font-dataset-npz/class_embeddings.npy --port 8080`  

To run all of these commands in sequence: `python test_e2e.py` 

It creates a tiny dataset, preprocesses the dataset, trains a model for an epoch, creates class embeddings, and runs the frontend. If running correctly, it should create a new folder `test-data` with images images and train/test data, as well as saving a model and embeddings. Finally, you should be go to the frontend at `localhost:8080`, upload one of the test images, and verify there are no errors. Note that we aren't expecting the model to actually perform well, but everything else should be working.

# Overview  

**Problem Statement**: If you see a font in the wild and want to use it yourself, how do you know the name of the font? If its a paid font, how do you find similar free fonts? The goal of this project is to build a tool that can take an image of a font and return the most similar fonts from a dataset of free fonts. We will do this in three steps: 1) generating the dataset of different fonts, 2) training a model to learn the features of each font, and 3) building a frontend to take in an image with some text and return the most similar fonts.


## Main Software Pieces
- dataset generation
    - html template to render the text into fonts
    - flask server to put the baked html onto a webpage
    - selenium scraper to takes screenshots of those webpages
    - script to orchestrate this and save the data to disk as npz or pkl files
- model training and font embedding creation
    - model architecture file
    - dataset file to load dataset into pytorch DataLoader
    - training file to handle the training loop, evaluation, metrics
    - create class-average font embeddings using the trained model and dataset
- frontend
    - interface to take upload image, run it through the embedding model, do cosine similarity against and return the most similar fonts


### Example file structure
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
    - font-images-npz
        - label_mapping.npy ~1kb
        - train.npz - few MB
        - test.npz ~500kb
        - fontCNN_BS64-ED128-IC16.pt ~5mb
        - class_embeddings.test ~25kb

    - 

# Workflow

## Data generation
TODO retest data generation with the font_size and line_height args
- generate data: `python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/full_fonts_list.txt --output_dir data/font-images --samples_per_class 100 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  
TODO output_dir = input_image_dir = "data/font-images"  
- prep data: `python data_generation/prep_train_test_data.py --input_image_dir data/font-images --output_dir data/font-dataset-npz --test_size .1`

## Train Model and Generate Embeddings
TODO output_dir = data_dir = data/font-dataset-npz
- train: `python ml/train.py --data_dir data/font-dataset-npz --epochs 30 --batch_size 64 --learning_rate 0.0001 --weight_decay 0.01 --embedding_dim 128 --resolution 64 --initial_channels 16`

- once you have a trained model, create embeddings: `python create_embeddings.py --model_path fontCNN_BS64-ED512-IC16.pt --data_dir data/font-dataset-npz --output_path class_embeddings_512.npy`
TODO model_path = model_path = 'fontCNN_BS64-ED512-IC16.pt'
data_dir = data_dir = 'data/font-dataset-npz'  

## Frontend
TODO make sure I shouldn't be passing a different embeddings file
- launch frontend `python frontend_app.py --model_path fontCNN_BS64-ED512-IC16.pt --data_dir data/font-dataset-npz --embedding_file class_embeddings.npy --port 8080`


Note: when running the frontend, the model pt passed in needs to have an embedding dimension that matches the class embeddings. The model file has the embedding dimension in the name. For example, fontCNN_BS64-ED512-IC16.pt has an embedding dimension of 512.


## Machine learning 
### Inference
Our goal is to find which fonts are most similar to the unknown input font, so we need to have some idea of what all the *known* fonts look like in feature space and return the closest ones. 

To find that, we can take the trained model and for each class, average the output embedding over all the training examples to give us a prototype average for each class. 

For each font class (out of the ~700 fonts), we're:
- Taking all images of that font (1000 images per class in this case)
- Running each through the model to get its 1024-dimensional feature vector
- Computing the average of all these vectors for that class

So if Font_A with n images, we get one 1024-dim vector representing the "average characteristics" of Font_A


## TODO
 
[ ] model experimentation: data augmentation - vary positioning/layout, font size and style, color and background color, text itself  
[ ] try a clip model of same font different font?  
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5  
[ ] do we get anything out of top eigenvectors of the data covariance matrix  
[ ] distance between their mean images 
[ X ] how can i make the model name legible / get returned from the train script? the filename itself has some of the hyperparams baked in - for example `fontCNN_BS64-ED512-IC16.pt`  
- solution: create a ml.utils file with get_model_path() a

Idea - what if I just generated the iamges of all the charcters in PIL and then do all the data augmentation to the images where each image has just one character in it

### ML Questions
What if I framed my problem also as first a character recognition detection problem, and then used the sum of these 
[ ] find a good open-source ocr character segmentation model and use it to generate 
-> does a segmentation model help at all here? forcing the model to learn exactly which pixels are and aren't part of the font? caveat is that the low resolution images probably won't work

[ ] Are there any ML strategies for doing CCE on a dataset with a large number of classes?

[ ] how many datapoints per class do I want if I have around 700 classes? cifar1000 archs probably a good place to start
[ ] Is it better just to keep the classifier and return the top 5 classes or to omit the classifier and just use the get_embeddings() part of the model to extract the features and then compare that to the average features of each class?


# TODO
[ ] figure out how to incorporate weight and width into the data augmentation and perhaps the model explicitly ala https://fonts.google.com/specimen/Roboto/tester  
-> might involve downloading all the font files and rendering them differently  
[ ]  update the data generation to something less crude than this original ugly hack - text = text * 10  # Repeat text to ensure enough content  
[ X ] get the script reading the text from the lorum_ipsom.txt file  
[ X ] get the fonts read in to the html page from fonts.txt  
[ X ] get the script to render the text on an html page in the correct font  
[ X ] figure out how much data I need - 100-10k images per class  
[ X ] get the screenshots saving with minimal overlap  
[ X ] collect a dataset of screenshots  
[ X ] get the full sized model to train in colab on an A100  
    - taking ~90/s per iteration  
Make sure that the metrics look right and wandb   
[  ] Get a sweep working on colab 
[ ] Put a little demo of each of the fonts on the frontend 

[ X ] get the flask app to launch for the frontend using the model and class embeddings
[ ] make a simple frontend that is capable of taking in images of fonts and returning out similar fonts  
[ ] make a pretty frontend - maybe pay someone or ask conor  
[ ] start generating data augmentations  
 
[ ] make the validation dataset smaller  
[ ] add back model saving during the epochs instead of just at the end. i think there is a bug rn such that it saves the network state after the last epoch regardless of whether that's the best model or not.  
[ ] write a script to run to generate class embeddings using the training data after generating and saving the best train model so i can have multiple models saved and each of them can generate their own class embeddings  
[ ] figure out what data format to store the class embeddings in so that they can be used inside a flask app   
[ ] figure out how a flask app is supposed to work  
[ ] once the website it live, would be nice to be able to save the font images people are uploading so we can get a better idea of what kind of data augmentations to do


# Training experiments
First idea is just to get the loss to go down for train and test over a 30 epoch run with a tiny 3M parameter network.

# ML steps and ideas
[ ] train a distance model on cifar to make sure the idea works
[ ] figure out how to get the mean image of a class
[ ] how many examples per class do we need

## Metrics TODO
[ ] remove or debug empty classes metric
[ ] remove acc std
[ ] do I need step and epoch time charts?

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
-"Metric learning: focuses on learning distance metrics between data points"

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

I have 500 paragraphs, at an average of 10 lines each, thats 5000 lines or 5000*36 ≈ 180,000 pixels worth of text, which is about half of what we were shooting for. The code should loop back to the beginning at this point but that's fine

At an average 

try some different text sizes for data augmentation

TODO move more info about data generation to the README inside data_generation
and link to it here 
[Dataset Generation Readme](data_generation/README.md)
TODO link to a huggingface download section 




[ ] get an input-independent baseline by zeroing out inputs and seeing how it performs   
[ ] overfit on one batch, launch it and make sure it works on the frontend too  


# Project Notes
## Notes
## Experiments
## Challenges
## TODOs




## Questions
How much does it matter if the model is trained on jpg images but someone inputs a png image? The png in theory is not lossy, but we will have to resize it to the same size as the training images, which will introduce some lossiness.




TODO make it so people can upload images of paid fonts to find most similar free fonts.
- normalize the image to the same size
- ask people to type on a blank page with a certain font size to normalize?
- allow people to 

allow people to pay to get their font included - upload your font file and we generate the lorum ipsum

# TODO
how do i want to handle the labels?
- want to create image / label pairs 
do we want to split the data into train/test beforehand and store them in separate folders?
- maybe just copy the cifar10 structure 
- unpickle the objects and load into a dictionary with d{data: np.array(), labels}

in train.py, we want to function to generate_batches():
- download the dataset tar to the machine from huggingface if it's not on there
 
Pickling process:
-Read through your directory structure
-Create a label mapping (font name → integer)
-Load images and convert them to arrays
-Save everything in pickle format

Pickling challenges
-was reading in the data as 3 channel by default when I really have black and white images
-the images are already compressed when they got saved to the dataset. Reading them in with numpy uncompresses them and I failed to recompress the first couple attempts
    - two options: numpy built in compression or more aggressive compression with gzip and pickle 


## Compression
How are we handling compression?
What does image compression actually mean? Explain why it would be nonsensical to 

**On disk** We want the images compressed on disk so they don't take up too much space. If we were using cloud storage like AWS S3, having the images compressed will save money, since they charge based on the size of outgoing data rather than the number of transfer events.

**At train time**, we want the data uncompressed on the GPU so that we have arrays 

What kind of color space do we need for the training images?


Imagine the space of image augmentations that would be helpful if we wanted to let people upload arbitrary photos of fonts?
- font size
- bold / italicized
- font color, background color
- background texture
- text placement / centering

.npz is for saving multiple numpy arrays, .npy is for saving single arrays

.npz is a good choice here since it's more memory-efficient than .pkl for large arrays, and since the dataset is significantly larger than CIFAR.




-TODO some explaining about high dimensional representations where all the datapoints are super far from each other so being close in one dimension ends up being close in 

## Questions
[ ] does it matter whether you take the class average over the train set vs the validation or test set?  



# Misc
- try uploading to huggingface again

Do I need to do any regularization?



figure out how to combine steps in one script for gathering data and training the model
1) generate the data, save the data, prep_train_test_data.py     
2) train, generate class embeddings, save class embeddings, launch flask app  



# Similar tools
The first 3 google results for font finders failed to recognize any of the font images I uploaded. These were high quality 256x256 images, much better quality than I'm training on. They all seem to segment the characters individually and then match against the characters. 

I also tried a "Font Identifier AI" in the ChatGPT store that failed to recognize any of the fonts images I tried.  

The Font Finder Chrome plugin seems quite promising since it uses the information from the browser's elements / CSS. However the format it returns in isn't all that helpful: font-family (stack)

![Font Plugin Example](font-plugin.png)
https://chromewebstore.google.com/detail/font-finder/bhiichidigehdgphoambhjbekalahgha


----
[ ] TODO delete extra create_embeddings.py file inside ml (or figure out which one is useful)
[ X ] Write 5  tests that correspond to the 5 main files that create the data, train, and do the frontend
[ X ] Disable wandb when im running the unit tests, unless needed
[ ] TODO make sure the test command args in the readme correspond to the ones in teste2e
[ ] TODO add test time fixed dataset visualizations so we can concretely see how the model is predicting

[ X ] create some test scripts to create a few images per class, train 1 epoch and save the model, load the model and do inference
- need a way to pass around class embedding file names