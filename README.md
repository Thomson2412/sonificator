# sonificator

### Dependencies
* python3.9
* ffmpeg
* supercollider
* supercollider-language
* preferably cuda

### Install
```
python3.9 -m venv venv
source venv/bin/activate
pip install opencv-python
pip install opencv-contrib-python
pip install tensorflow
pip install torch==1.10 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install librosa
pip install pillow
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

