# SeaNos

 <p align="center">
<img src="https://github.com/bramiozo/SeaNos/blob/main/seannos.png" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>


The SeaNos pipeline was [part of an art exhibition at the International Documentary Festival Amsterdam](https://festival.idfa.nl/film/18d83eed-a061-4441-86bd-bef7c1a407eb/you-can-sing-me-on-my-way/) (IDFA).

Development of ```Text``` $\rightarrow$ ```Sean Nos``` using existing models.

Tutorials: [HF](https://huggingface.co/learn/audio-course/chapter6/tts_datasets)

```Text``` $\rightarrow$ ```Speech mel spectrogram``` with TTS $\rightarrow$ Waveform with vocoder


# Technologies text-to-speech

## **TTS with MMS**

[TTS](https://huggingface.co/coqui/XTTS-v2), [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms), [MMS coverage](https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html)


Download the Irish model
```wget https://dl.fbaipublicfiles.com/mms/tts/gle.tar.gz # Irish gle)```
```wget https://dl.fbaipublicfiles.com/mms/tts/cym.tar.gz # Welsh gle)```

```pip install TTS ```
```python

from TTS.api import TTS
tts = TTS("loc/to/irish/model", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")
```

## Suno

[Suno](https://huggingface.co/suno/bark)
```python
from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-speech", "suno/bark")

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"do_sample": True})

scipy.io.wavfile.write("bark_out.wav", rate=speech["sampling_rate"], data=speech["audio"])
```
## Espnet
[Espnet](https://github.com/espnet/espnet)


## Open private models:
https://huggingface.co/ylacombe/vits_vctk_irish_male
https://huggingface.co/ylacombe/vits_ljs_irish_male


## Voice cloning

TacoTron 2



# Examples

```
An aimsir bhog, bhlàth, sa ghàrradh,
Far a bheil dealain-dè a’ dannsa gu h-àrd;
Càirdeas nan speur is blàths an t-samhraidh,
A’ cur cridheachan ann an gaol mar aon.
```


# Architecture


 <p align="center">
<img src="https://github.com/bramiozo/SeaNos/blob/main/SeanNos%20-%20Architecture.png" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>

Training pipeline:
```raw wav snippets | align with | raw text snippets```

```raw text snippets $\rightarrow$ phonetics```

``` raw wav snippets ``` $\rightarrow$ ```clean wav snippets``` $\rightarrow$ ```pairs(phonetics, wav)``` $\rightarrow$ ```text-to-speech``` $\rightarrow$ ```Sean Nos``` 

Inference pipeline:
```text generation``` $\rightarrow$ ```text-to-speech``` $\rightarrow$ ```Sean Nos```

## Production

* SQLite3: ```timestamp, news original, news summary, lyrics, Settings hash, TTS model hash, SS model hash, LLM version```
* TTS server to serve low-latency TTS
* TTS server to serve low-latency Speech-to-Song
* RSS news fetcher with SQL fallback
* LLM API with SQL fallback

# References

* https://huggingface.co/facebook/seamless-streaming
* https://huggingface.co/audo/seamless-m4t-v2-large
* https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f
* https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/g2p.html
* https://www.focloir.ie/en/page/tools.html
* [training](https://docs.coqui.ai/en/dev/tutorial_for_nervous_beginners.html)
* [training2](https://docs.coqui.ai/en/dev/finetuning.html)
* [speech-to-song](https://arxiv.org/abs/2110.06546)

# Installation:

```sudo apt install ffmpeg```
```sudo apt install espeak-ng```
```sudo apt install vlc```
```sudo apt install ubuntu-restricted-extras```



