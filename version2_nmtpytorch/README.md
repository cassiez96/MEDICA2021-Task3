# MEDIQA2021 Task 3 using nmtpytorch

### Download Task 3 Data (report and xray images)
Go here : https://drive.google.com/drive/u/1/folders/1zqu7i3Pm2A4kcdeLO9d1TXQYG8UaAbnQ<br/>
Download train, dev and indiana_dev.json<br/>
Download indiana_images.zip, put it in "indiana_images" folder<br/>
Put mimic-cxr images in "images" folder<br/>

### Split data (findings, background+findings, impression)
Run:<br/>
`python split_input_target.py`<br/>
Files are created in out_nmtpytorch folder (or here https://drive.google.com/drive/u/1/folders/1ATUSX3o9vhgQuHOe18WgRobftOyz3zu0)

### Tokenize data
To tokenize files (but dont do that yet), do :

```
for split in train dev indiana_dev; do
    for mode in findings bg_and_findings impression; do
      for llang in en; do
        moses_scripts/lowercase.perl < out_nmtpytorch/${split}.${mode} | moses_scripts/normalize-punctuation.perl -l $llang | \
          moses_scripts/tokenizer.perl -q -a -l $llang -threads 4 > out_nmtpytorch/${split}.${mode}.lc.norm.tok
      done
    done
done
```

### Generate vocabulary
First, download nmtpytorch: https://github.com/lium-lst/nmtpytorch
Go to the directory `nmtpytorch/bin` and execute:

```
chmod u+x "nmtpytorch/bin/nmtpy-build-vocab"
cd nmtpytorch/bin
./nmtpy-build-vocab -o out_nmtpytorch out_nmtpytorch/train.findings out_nmtpytorch/train.impression out_nmtpytorch/train.bg_and_findings
```

### Install and configure dependencies for nmtpytorch (required on Google Colab)
```
python3 -m pip install --upgrade pip
pip install nmtpytorch
python3 -m pip install --upgrade Pillow
cd nmtpytorch/bin
chmod u+x nmtpy-install-extra
./nmtpy-install-extra
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl # default torch version not compatible with Google Colab GPU
```

### Train nmtpytorch
Run nmtpytorch with configuration file: `nmtpy train -C mmt-task-fd-impr-encdecinit.conf`



