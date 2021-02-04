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
!cd nmtpytorch/ && python setup.py develop && \
nmtpy-install-extra 
```
<b>Create vocab</b><br/>

```
nmtpy_dir=preprocessing/out_nmtpytorch
for file in train.findings.tok train.impression.tok  train.bg_and_findings.tok
do
    nmtpy-build-vocab -o ${nmtpy_dir} ${nmtpy_dir}/${file}
done
```

<b>Train nmtpytorch</b><br/>
Run nmtpytorch with configuration file: `nmtpy train -C baseline_avgpool.conf`


