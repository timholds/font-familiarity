# TODO quickstart
Create a new virtual environment and install the requirements
```
python3 -m venv font-env
source font-env/bin/activate
pip install -r requirements.txt
```

# Overview 
The readme is a project notes section for now.

**Problem Statement**: If you see a font in the wild and want to use it yourself, how do you know the name of the font? If its a paid font, how do you find similar free fonts? The goal of this project is to build a tool that can take an image of a font and return the most similar fonts from a dataset of free fonts. We will do this in three steps: 1) generating the dataset of different fonts, 2) training a model to learn the features of each font, and 3) building a frontend to take in an image with some text and return the most similar fonts.

## Main Software Pieces
- html template to render the text into fonts
- flask server to put the baked html onto a webpage
- selenium scraper to takes screenshots of those webpages
- (optional) compressing the data for storage 
- saving the data to disk as npz or pkl files
- model architecture file
- file to load dataset into pytorch DataLoader
- file to handle the training loop, evaluation, metrics
- inference and calculating the most similar fonts, creating the font embeddings
- frontend to take in an image and return the most similar fonts


# Machine learning 

## Steps
Simple baseline model
Get a model to overfit
Regularize
Try a more complex model
Try a more complex model with more data
Try a more complex model with more data and more regularization
Try a more complex model with more data and more regularization and more data augmentation
[ ] Visualize model predictions some small holdout batch every epoch

## TODO
[ ] critical:  there's an off by one error in the labels mapping. When I try to use a super overfit model on the frontend, it returns the wrong class with super high certainty! Checking this out closer, I notice it's predicting the class 1 off from the correct class. For example, on an image of the "yesseva" font, the model predicts "yesteryear" with 99.5% certainty.  
[ ] model experimentation: data augmentation - vary positioning/layout, font size and style, color and background color, text itself  
[ ] try a clip model of same font different font?  
[ ] train a classifier model and use the average class features to find which classes are closer or more similar to each other and return the top 5  
[ ] do we get anything out of top eigenvectors of the data covariance matrix  
[ ] distance between their mean images  

Idea - what if I just generated the iamges of all the charcters in PIL and then do all the data augmentation to the images where each image has just one character in it

## Quetions
What if I framed my problem also as first a character recognition detection problem, and then used the sum of these 
[ ] find a good open-source ocr character segmentation model and use it to generate 
-> does a segmentation model help at all here? forcing the model to learn exactly which pixels are and aren't part of the font? caveat is that the low resolution images probably won't work

[ ] Are there any ML strategies for doing CCE on a dataset with a large number of classes?


# questions
- how many datapoints per class do I want if I have around 700 classes? cifar1000 archs probably a good place to start
- Is it better just to keep the classifier and return the top 5 classes or to omit the classifier and just use the get_embeddings() part of the model to extract the features and then compare that to the average features of each class?


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

# Process
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


[ ] get an input-independent baseline by zeroing out inputs and seeing how it performs   
[ ] overfit on one batch, launch it and make sure it works on the frontend too  


## Questions
How much does it matter if the model is trained on jpg images but someone inputs a png image? The png in theory is not lossy, but we will have to resize it to the same size as the training images, which will introduce some lossiness.

## Image parameters vs filesize 
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


ideally we want to add the downstream ML objective to this table since that is the actual tradeoff we care about 

what if we increase the font size instead of increasing the resolution / amount of text in each image

some blurring can actually be desirable for the ML task as it has a regularizing effect. in other words, blurring helps bias the parameters of the network to be closer to 0, which will lead to smoother loss landscapes and better generalization. in other words, small weights and biases mean less expressability but smoother interpolation between training points. this all reduces overfitting. 


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


# Compression
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

# Experiments
[ ] Try contrastive loss
[ ] Try triplet loss
[ ] Experiment with weight decay
[ ] Test single conv layers vs back to back conv layers between downsamples
[ ] Add a wandb config file and sweep conv sizes
[ ] Compare my model to a LoRA of gemmapali or a multimodal llama

# TODO 
[ ] move the class embedding generation into prep train test data.py? Otherwise go back to using a class embedding argument in the frontend_app.py instead of searching for it in the dataset folder 
[ ] fix the error when uploading an image
[ ] add some processing to resize images coming in and make them single channel
[ ] remove the model saving part for the sweep. once we know which model trains best we can retrain it and reanble the saving.
[ ] cleanup the metrics 
- [ ] is the epoch time metric actually telling me how long an epoch takes?


[X] Figure out how to keep track of the model experiments - use weights and biases 

Test my model on cifar dataset instead to sanity check that it is capable of learning anything

[ ] Add a note about how the data is stored in the npz and the best way to access it
- remove "Keys in NPZ file: ['images', 'labels']"

[X] create a small test dataset for iterating on the model - font_dataset_npz_test

[X] make it so that it only saves the class averages at the end of training instead of everytime a new best model gets saved
[X] add a LR warmup and cosine learning rate
[ ] clean up hardcoded flatten_dim
        self.flatten_dim = 128 * 4 * 4 
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),


# Inference
Our goal is to find which fonts are most similar to the unknown input font, so we need to have some idea of what all the *known* fonts look like in feature space and return the closest ones. 

To find that, we can take the trained model and for each class, average the output embedding over all the training examples to give us a prototype average for each class. 

For each font class (out of the ~700 fonts), we're:
- Taking all images of that font (1000 images per class in this case)
- Running each through the model to get its 1024-dimensional feature vector
- Computing the average of all these vectors for that class

So if Font_A with n images, we get one 1024-dim vector representing the "average characteristics" of Font_A


-TODO some explaining about high dimensional representations where all the datapoints are super far from each other so being close in one dimension ends up being close in 
-TODO does it matter whether you take the class average over the train set vs the validation or test set?


# Misc
what is the best way to store this dataset if I wanted to make it uber scalable? Store it on AWS S3 as compressed
-Add code to download dataset from a public bucket
- try uploading to huggingface again

Do I need to do any regularization?

metrics are a clusterfuck right now. i really want probably 12 metrics right now, the most important being test validation 

Establish a baaaaseline baseline with a one layer MLP
Establish simple conv baseline for the encoder
Test a resnet as the encoder
Test a resnet with supervised contrastive loss (does it need to be a pretrain)

figure out how to combine steps in one script for gathering data and training the model
1) generate the data, save the data, prep_train_test_data.py     
2) train, generate class embeddings, save class embeddings, launch flask app  



# Similar tools
The first 3 google results for font finders failed to recognize any of the font images I uploaded. These were high quality 256x256 images, much better quality than I'm training on. They all seem to segment the characters individually and then match against the characters. 

I also tried a "Font Identifier AI" in the ChatGPT store that failed to recognize any of the fonts images I tried.  

The Font Finder Chrome plugin seems quite promising since it uses the information from the browser's elements / CSS. However the format it returns in isn't all that helpful: font-family (stack)

![Font Plugin Example](font-plugin.png)
https://chromewebstore.google.com/detail/font-finder/bhiichidigehdgphoambhjbekalahgha

TODO delete extra create_embeddings.py file inside ml (or figure out which one is useful)

# Test workflow
Same as the regular workflow but intended to be used as an endtoend test. It creates a tiny dataset, preprocesses the dataset, trains a model for an epoch, creates class embeddings, and runs the frontend. We aren't expecting the model to actually perform well, but everything else should be working. Namely, the 

# Workflow
- generate data: `python data_generation/create_font_images.py --text_file data_generation/lorem_ipsum.txt --font_file data_generation/fonts.txt --output_dir font-images3 --samples_per_class 100 --image_resolution 128 --port 5100 --font_size 35 --line_height 1.5`  
- prep data: `python data_generation/prep_train_test_data.py`
- train: `python ml/train.py --data_dir "font_dataset_npz" --embedding_dim 256 --batch_size 32`
- once you have a trained, saved pt model, use it to create embeddings: `python create_embeddings.py --model_path fontCNN_BS64-ED512-IC16.pt --data_dir font_dataset_npz/ --output_path class_embeddings_512.npy`
- launch frontend


Note: when running the frontend, the model pt passed in needs to have an embedding dimension that matches the class embeddings. The model file has the embedding dimension in the name. For example, fontCNN_BS64-ED512-IC16.pt has an embedding dimension of 512.

TODO add test time fixed dataset visualizations so we can concretely see how the model is predicting

TODO create some test scripts to create a few images per class, train 1 epoch and save the model, load the model and do inference
- need a way to pass around class embedding file names