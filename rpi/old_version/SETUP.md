# Software Setup

This document describes the software setup required on a Raspberry Pi to run
CapuchinAI.
Following the instructions in this document from this directory yields the setup
used in our experiments.

## Install Miniconda

We recommend using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)
for virtual environments and package management. Miniconda is a lightweight
version of Anaconda, and we have found it runs well on a Raspberry Pi.
To install miniconda, execute the following commands.
```
mkdir ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda_install.sh
bash ~/miniconda3/miniconda_install.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
```
Then restart the terminal window.

## Create a New Conda Environment

Create a conda environment by running
```
conda create -n capuchinai python=3.10 -y
```
and then activate it as follows.
```
conda activate capuchinai
```

## Install prerequisites

With the conda environment activated, install the modules required to run our
model using the following command.
```
pip install -r yolo_requirements.txt
```

## Run CapuchinAI

To run CapuchinAI, first download your trained model weights on the Raspberry Pi.
Place them in this directory and name the file `best.pt`.
With the conda environment activated, run
```
python capuchin_recorder_headless.py & PID1=$!
```
This should start recording in the background.
At the same time, we run the GPIO reward script using the system Python
distribution.
```
sudo /usr/bin/python3 touchscreen_reward_interface.py
```

When finished, kill the YOLO process if it is still running.
```
kill $PID1
```

