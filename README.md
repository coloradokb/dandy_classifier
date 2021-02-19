This project is a very simple binary image classification model for me to do some "real world learning":
    - Dandelions
    - Anything NOT dandelions :)

Initial thoughts and findings:
    Need lots and lots more images. While the training results in decent accuracy, the validation loss is substantial. My initial 1,200+ images (50% dandelion/50% not) seems woefully small. May need upwards of 10,000+ of each....maybe. Initially the focus was not on dandelions that have already flowered, but more on just the leaves being the more prominent feature. Next spring/summer, I'll need to diversify the dandelions.
    I had initially started with 'Square' images to keep it uniform. However, with help from the 'kids,' in the neighborhood I ended up allowing all sizes and shapes. Seems that to get well over 10,000 of each, that's a restriction I'll have to work through.
    
Model training is done in two files:
1. dandy_classifier.ipynb - notebook for general use/training/learning. I'll usually take this and in turn create the script for running separately when needed.
2. dandy_classifier.py - script that will be used in automation tasks (re-running of model training late night after new images have come in, possibly)
3. dandy_predictor.ipynb - notebook to try agains unseen, different images
4. model_builder_resnet.ipynb - notebook that I'm just starting to play with using transfer learning as I really may never get enough images :<|

**Images: filesize of Images.zip is too large for github. Go to this url (https://www.kaggle.com/coloradokb/dandelionimages) on Kaggle.com for the dataset to download**

Directory structure and contents for the future when I can spend more time on it!
|--metrics:: simple text/csv files that give metrics information on last training. (;) is default delimmiter
|
|--saved_models:: if last training is > 75% accuracy (not val_acc) then save model for later predictions. Make "copy" if it's one to set aside. This has numerous
|                 models where I played around with other resnet models.
|
|--training_checkpoints:: sometimes training time is long. save checkpoints in case of err so we can pick up again from last check
|
|--Images:: Directory that has Dandelions and non-dandelion pictures. Currently a little over 1,000 pics total. Most pics are high res from a few different phones.               
            Total size at initial check-in is 3.6GB.
            Images/dandelion have images that contain a dandelion
            Images/other do not...often just a snapshot of grass
