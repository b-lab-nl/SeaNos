# SeaNos

 <p align="center">
<img src="https://github.com/bramiozo/SeaNos/blob/main/seannos.png" alt="image" width="300" height="auto" >
 </p>
<hr width=100%>


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

Training pipeline:
```(raw wav)``` $\rightarrow$ ```clean wav``` $\rightarrow$ ```(text, wav)``` $\rightarrow$ ```text-to-speech``` $\rightarrow$ ```Sean Nos``` 

Inference pipeline:
```text generation``` $\rightarrow$ ```text-to-speech``` $\rightarrow$ ```Sean Nos```