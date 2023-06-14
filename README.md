# GrooveGenie-A-copyright-free-music-generator

## Abstract
Our project aims to address the challenge of using copyrighted music in social media content creation. Creators often face copyright issues when using music in their videos, podcasts, and other social media content, especially for intro/outro beats or mid-video breaks to keep their audience engaged. To avoid copyright infringement, creators have to either purchase a license to use music or create their own. However, these options can be time-consuming and costly and may not necessarily fit the specific requirements of the content. Therefore, we developed Groovie-Genie, which will generate loopable 30 seconds of audio for the influencer to use, which is perfect for intro/outro beats for podcasts, YouTube videos, Instagram videos, and generally all social media posts. We will discuss the three different approaches we took in detail and also the challenges we faced.


## More info
Read our report that explains the motivation, the three process we used, and everything in detail [here](./Docs/285_Project_GroovieGenie_Final_Report.pdf)

For a brief overview, refer our ppt [here](./Docs/CSE_285_Groovie_Genie_ppt.pptx)


## Folder Structure and How to run
The folder names are pretty explanatory.

There exists readme inside each folder explaining how to run.

# Outputs

You can listen to the generated music [here](https://drive.google.com/drive/folders/10CUluH9OP9jbVhLTVSuWvnCOsqMnsXLc?usp=sharing). These are mp3 files converted from the generated midi files.

If you prefer to see the actual generated midi files, it's inside the music_metrics and the respective folders. You can use the sound font file (available inside Music_Metrics folder) to listen to it if your OS doesn't play mid files by default. Just google play mid files using custom Sound font.


#### Requirements.txt
torch2.0 should definitly work for methodology 2, 3.
if it gives error in 1, try degrading,, there was a bug in the training loop not allowing multiple dataloader threads that we fixed in methodology 2 and 3.

Believe me we would have loved to give a requirements.txt but it's a mess on our end too... :_)

P.S.: Some Specific libraries used will be mentioned in the readme inside the folders.


### Feel free to reach out to us for any questions you may have.


## Authors
- [@Jay Jhaveri](https://github.com/JayJhaveri1906)
- [@Andrew Ghafari](https://github.com/AGhafaryy)