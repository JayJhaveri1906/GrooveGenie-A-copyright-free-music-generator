# Methodology 1: Using Facebook's EnCodec + Bert for Genre.

Read explanation in report.

## How to run

### Preprocessing
Before executing main.py, you should preprocess the mp3 files.

Download the mp3 files from the FMA dataset. Here we are using fma-small.

Open audio_preprocessing.py

Inside the if __name__ == __main__:
Change the excel file location to your path. (Line 82)

Make sure the excel file is structured the way it is structured here.

Rest should be pretty intuituve.


#### P.S. in our experience, transformers==4.8.0 usually works the best without any errors.


### Running Main.py

First open constants.py

If this is your first time training, you should change the variable PRETRAINED to False.

Once, you set your parameters in the constants.py, you can run the main.py file. It should automatically import the CRNN Gan architecture and start running. (You may choose to run normal GAN, by just changing the import from lstmGAN to GAN in the utils file.

It will auto create a runs folder, and unique run folder within it.

Will store the outputs generated.

##### ALSO NOTE: IF RUNNING ON WINDOWS: WHILE GENERATING and SAVING MUSIC, windows pytorch doesn't support "mp3" files, no idea why. Do ".wav" instead.


