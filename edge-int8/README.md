# rslt3 int8 edge weights

This directory contains the weight-only int8 `rslt3` checkpoint and the two local feature/tokenizer configuration files required for inference. It is separate from `checkpoints/rslt3`; the paper and the website use the float32 checkpoint.

## Score on the held-out OpenSLR clips

The 100 held-out archive-f clips were decoded using the same feature extractor, tokenizer, beam width five, and NFC scorer as the float32 run:

| Weights | Clips | CER | WER | RTF |
|:--|--:|--:|--:|--:|
| rslt3 float32 | 100 | 0.165289256 | 0.427672956 | 0.078 |
| rslt3 int8 weights | 100 | 0.165289256 | 0.430817610 | 0.068 |

CER exactly matches the stored float32 result at the reported precision; WER is slightly higher. The int8 run's RTF was measured on the current machine and should not be directly compared with the stored float32 RTF from its original evaluation environment. The generated int8 file is for a small computer or older laptop. ONNX export and ONNX Runtime decoding have not been implemented or verified, so this is not a native phone model and does not establish suitability for every edge device. The browser microphone page remains the phone path.

Run from the repository root:

```bash
python3 runtime/transcribe.py --model edge-int8 --audio clip.wav
```

The command uses FFmpeg for 16 kHz mono conversion, caps input at 30 seconds, and selects `student.int8.pt` automatically. To recreate the checkpoint from the unchanged float32 source:

```bash
python3 runtime/pack_int8.py --model checkpoints/rslt3 --output edge-int8/student.int8.pt
```
