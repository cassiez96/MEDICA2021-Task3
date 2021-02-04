# MEDIQA2021 Task 3 using nmtpytorch

### Download Task 3 Data (report and xray images)
Go here : https://drive.google.com/drive/u/1/folders/1zqu7i3Pm2A4kcdeLO9d1TXQYG8UaAbnQ<br/>
Download train, dev and indiana_dev.json<br/>
Download indiana_images.zip, put it in "indiana_images" folder<br/>
Put mimic-cxr images in "images" folder<br/>

### Split data and extract image features(findings, background+findings, impression)
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

### Tokenize data (Stanford NLP Tokenizer)
This is the tokenizer to use

```
python stanford_tokenizer.py
```

### nmtpytorch
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



