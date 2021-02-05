```
baseline_avgpool.conf
    --> This is model: simplemmt-r1b8d6
    --> Best LOSS so far: 1.49 @ validation 15
    --> Best BLEU so far: 31.27 @ validation 28
    --> Best METEOR so far: 26.68 @ validation 28
    --> Best ROUGE so far: 61.48 @ validation 20
    Training finished on 04-02-2021 19:25
```

```
baseline_mono.conf
    --> This is model: nmt-r9fa7d
    --> Best LOSS so far: 1.49 @ validation 14
    --> Best BLEU so far: 31.51 @ validation 31
    --> Best METEOR so far: 26.93 @ validation 30
    --> Best ROUGE so far: 61.48 @ validation 22
    Training finished on 04-02-2021 19:33
```

So far mono is better than multimodal.
Need to improve results.

try ctxmul:
```
--> This is model: simplemmt-re5e4a
--> Best LOSS so far: 1.49 @ validation 17
--> Best BLEU so far: 31.06 @ validation 32
--> Best METEOR so far: 26.51 @ validation 32
--> Best ROUGE so far: 61.53 @ validation 28
Training finished on 05-02-2021 14:32

```
doesnt change much