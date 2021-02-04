# MEDIQA2021 Task 3 using nmtpytorch

### Preprocessing folder

#### Download
Go here : https://drive.google.com/drive/u/1/folders/1zqu7i3Pm2A4kcdeLO9d1TXQYG8UaAbnQ <br/>
Download train, dev and indiana_dev.json<br/>
Download indiana_images.zip, put it in "indiana_images" folder<br/>
Put mimic-cxr images in "images" folder<br/>

#### tokenize
Run:<br/>
`python preprocess.py`<br/>
This script split sections and tokenize input files <br/>
Files are created in out_nmtpytorch folder (or here https://drive.google.com/drive/u/1/folders/1ATUSX3o9vhgQuHOe18WgRobftOyz3zu0)


### nmtpytorch folder
<b>On google colab do </b><br/>
```
!git clone https://github.com/lium-lst/nmtpytorch && \
cd nmtpytorch/ && python setup.py develop && \
nmtpy-install-extra 
```
<b>Create vocab</b><br/>
`nmtpy-build-vocab -o out_nmtpytorch out_nmtpytorch/train.findings.tok`

<b>Train nmtpytorch</b><br/>
Run nmtpytorch with configuration file: `nmtpy train -C mmt-task-fd-impr-encdecinit.conf` (edit `__FILE_PREFIX__` and `__SAVE_PATH__` based on `out_nmtpytorch`'s path prefix and where to save the trained model) 



