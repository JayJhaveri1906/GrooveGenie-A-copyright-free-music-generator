# Methodology 2: Using only CRNN GAN + Piano MIDI files.

Read explanation in report.

## How to run

VERY similar to method 1.

### Preprocessing
in preprocessing.py on line 52, change the dataPath to your folder path. (Make sure that folder only has .mid files, not even .gitignore)
Run it, it should create a .pth file and save it in according to the path mentioned on line 53.

### Running Main.py 

First open constants.py

If this is your first time training, you should change the variable PRETRAINED to False.

PREPROCESSED_DATA_PATH on line 13, should be changed to the path of the output of preprocessing.py

Here, unlike method 1, we store GEN and DIS weights inside one path called cRnnGan.pth (line 35, only applies when pretrained is True)


Once, you set your parameters in the constants.py, you can run the main.py file. 

##### NOTE: We also have wandb setup, so you should change the entity name in wandb.init line in the utils.py file. Refer to wandb quickstart docs if you are new to this. It should be pretty intuitive.

It should automatically import the CRNN Gan architecture and start running.

It will auto create a runs folder, and unique run folder within it.

Will store the outputs generated.

### Inference
You may directly run the genMusic.py file to generate 10 music files. Make sure pretraining is set to true in constants and the DISGEN_PATH on line 34 is to the model you want!

