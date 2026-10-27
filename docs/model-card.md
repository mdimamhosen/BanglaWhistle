---
license: cc-by-sa-4.0
language:
- bn
pipeline_tag: automatic-speech-recognition
tags:
- bangla
- bengali
- speech-recognition
- asr
- compact-model
---

# BanglaWhistle rslt3

A compact Bangla automatic speech recognition checkpoint. The model has **15,968,768 parameters** and the float32 checkpoint is **63,944,321 bytes**. This model card reports the saved `rslt3` run only.

## Evaluation

Lower CER and WER are better. OpenSLR 53 archive f is held out from training but belongs to the same corpus. FLEURS and Common Voice are external test sets.

| Test | Clips | rslt3 CER / WER | whisper-tiny-bn CER / WER | rslt3 RTF |
|:--|--:|--:|--:|--:|
| OpenSLR 53 zip f | 100 | 0.165 / 0.428 | 0.279 / 0.871 | 0.078 |
| FLEURS `bn_in` test | 920 | 0.659 / 0.909 | 0.408 / 0.833 | 0.273 |
| Common Voice 17 Bengali test | 9,327 | 0.538 / 0.939 | 0.331 / 0.862 | 0.207 |

The paired mean utterance-CER gap (rslt3 minus the baseline), with stored 95% bootstrap interval, is −0.133 [−0.174, −0.091] on OpenSLR, +0.260 [+0.251, +0.268] on FLEURS, and +0.184 [+0.181, +0.188] on Common Voice. The baseline is ahead on both external tests. RTF values are specific to the recorded evaluation environment and are not a service latency guarantee.

## Files

- `student.pt`: rslt3 float32 state dictionary.
- `bn.syllables.json`: Bangla syllable-piece vocabulary.
- `preprocessor_config.json`: Whisper-compatible 16 kHz feature extractor settings.
- `model/`: model and tokenizer definitions needed by the inference script.
- `runtime/transcribe.py`: single-file transcription CLI, beam width five, 30-second input cap.
- `scorer.py`: NFC normalization and CER/WER functions.
- `results.json`: saved evaluation artifact for this checkpoint. It contains transcript references and hypotheses but no audio files.

## Inference

Install Python 3.11 or newer, PyTorch, Transformers, NumPy, and FFmpeg, then run from this directory:

```bash
python3 runtime/transcribe.py --model . --audio recording.wav
```

The CLI converts to 16 kHz mono with FFmpeg, truncates input at 30 seconds, and uses beam width five. The model is designed for short, single-speaker Bangla speech. It does not provide diarization, timestamps, confidence scores, or translation.

## Training and data

The student was trained using OpenSLR 53 archives 0–7, with a frozen Whisper-small encoder as a training guide. The teacher is not loaded for inference and is not included in this repository. The run ended at the 12-hour Kaggle session limit. The tokenizer has syllable-like Bangla pieces; no tokenizer ablation was performed.

The held-out OpenSLR archive f, FLEURS test audio, and Common Voice test audio are not included here. Common Voice evaluation used the `fsicoli/common_voice_17_0` mirror identified in the results artifact. Review the source corpus terms and attribution requirements before redistributing this model or using it commercially. OpenSLR 53 lists its corpus under CC-BY-SA 4.0.

## Limitations

This checkpoint is not a general-purpose or state-of-the-art Bangla recognizer. Its lower error on a held-out OpenSLR archive does not carry over to FLEURS or Common Voice. Test sets also differ in collection and transcription conventions. The OpenSLR comparison uses 100 clips, and the run has no repeated seeds or tokenizer ablation. Validate on audio from the intended domain before relying on its output.

## Citation

If you use the training corpus, cite the OpenSLR 53 paper:

> O. Kjartansson, S. Sarin, K. Pipatsrisawat, M. Jansche, and L. Ha. “Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali.” SLTU, 2018. DOI: 10.21437/SLTU-2018-11.

The accompanying manuscript is `paper/paper.pdf` in the source repository.
