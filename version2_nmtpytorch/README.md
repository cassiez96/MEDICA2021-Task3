# MEDIQA2021 Task 3 using nmtpytorch

### Preprocessing folder

#### Download
Go here : https://drive.google.com/drive/u/1/folders/1zqu7i3Pm2A4kcdeLO9d1TXQYG8UaAbnQ <br/>
Download indiana-views.json and mimic-views.json and put it in data folder<br/>
Download indiana_images.zip, put it in "data/indiana_images" folder<br/>
Put mimic-cxr images in "data/mimic-cxr-images" folder<br/>

#### Make training files
Run:<br/>
`python one_image.py`<br/>
This script split sections and tokenize input files <br/>
One image just take one image per report. <br/> Feel free to make other version.
Files are created in out_nmtpytorch/one_image/ in this example

### nmtpytorch folder
<b>On google colab do </b><br/>
```
!cd nmtpytorch/ && python setup.py develop && \
nmtpy-install-extra 
```
<b>Create vocab</b><br/>

```
nmtpy_dir=preprocessing/out_nmtpytorch/one_image
for file in train.findings.tok train.impression.tok  train.bg_and_findings.tok
do
    nmtpy-build-vocab -o ${nmtpy_dir} ${nmtpy_dir}/${file}
done
```

<b>Train nmtpytorch</b><br/>
Run nmtpytorch with configuration file: `nmtpy train -C baseline_mono.conf`


