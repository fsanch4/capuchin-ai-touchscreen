# capuchin-ai-touchscreen

This repository contains code for the
[CapuchinAI](https://www.biorxiv.org/content/10.1101/2025.11.07.687266v2.full)
project, which aims to bring lab-based touchscreen cognitive testing methods to
wild primates using face recognition.
Specifically, this repository contains the code required to (a) train an
individual recognition model, and (b) run the live detection and reward
interface systems on a Raspberry Pi. Note that this repository does *not*
contain details pertaining to dataset preparation and hardware setup: please
refer to our journal article for those.

This repository contains the following components:
1. `CapuchinAI_train.ipynb`: An iPython notebook that downloads the "Multiple
    Capuchins" dataset and trains an individual recognition model using it. The
    notebook also contains instructions for using your own dataset. We recommend
    running the notebook on [Google Colab](https://colab.research.google.com).
2. `rpi/`: A directory containing code to be run on the Raspberry Pi, as well as
    software setup instructions.

