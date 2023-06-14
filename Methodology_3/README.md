# Methodology 2: Using only CRNN GAN + MultiInstrument PianoRolls.

Again read description in the report.

## How to run:

To run inference on Methodology 3 you need to keep this in mind:
1- When you are running the top cells (training cells), you need to have tha latest version of the pypianoroll library (1.0.4). You can use 
`pip show pypianoroll`
to make sure the versions match. 

2- After that, when you want to run inference, you need to downgrade the library to 0.5.3 using
`pip show pypianoroll==0.5.3`, 
and then run the latest cell that runs inference. Also use !pip show pypianoroll to make sure that it downgraded properly. And then run the inference cell and enjoy the music! (Try restarting the kernel before inference, if you run into any error during inference)

This is due to some dependencies in the libraries and some missing functions in the newest version that we couldn't resolve with the new version so we had to do it this way.