# ECG-Anomaly-Detection
ECG Anomaly Detection at the Edge using MAX78000
## Setup environment
1. install **pyenv**
   ```
   https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
   ```
2. Creating the Virtual Environment
   ```
   $ cd <your/project>
   $ git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
   $ cd ai8x-training
   $ pyenv local 3.11.8
   $ python -m venv .venv --prompt ai8x-training
   $ echo "*" > .venv/.gitignore
   $ source .venv/bin/activate
   (ai8x-training) $ pip3 install -U pip wheel setuptools
   (ai8x-training) $ pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
   (ai8x-training) $ pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/rocm5.7
   (ai8x-training) $ deactivate
   $ cd <your/project>
   $ git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git
   $ cd ai8x-synthesis
   $ pyenv local 3.11.8
   $ python -m venv .venv --prompt ai8x-synthesis
   $ echo "*" > .venv/.gitignore
   $ source .venv/bin/activate
   (ai8x-synthesis) $ pip3 install -U pip setuptools
   (ai8x-synthesis) $ pip3 install -r requirements.txt
   ```
## Overview
