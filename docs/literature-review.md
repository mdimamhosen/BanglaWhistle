# Literature notes for Section 2

These notes are for the related-work section of the BanglaWhistle paper. The number we own is character error 0.165 on 100 held-out clips from OpenSLR 53 archive f. Most published Bangla numbers use a different test, a different split, or word error instead of character error. Those numbers stay in the paper as descriptions of other work. They do not go in the same column as 0.165.

## The claim that fits the search

Bangla speech recognition papers exist in large numbers. Lightweight Bangla systems also exist. They are fine-tunes of Whisper Tiny, about 39 million parameters. A search of journal papers, preprints, and public model cards on 27 October 2026 did not find a peer-reviewed paper for a custom Bangla recognizer at about 16 million parameters. That is the gap. It is not a claim that Bangla research is missing.

## Papers and models

### Kjartansson et al., 2018

O. Kjartansson, S. Sarin, K. Pipatsrisawat, M. Jansche, and L. Ha, “Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali,” *SLTU*, 2018. Data: https://www.openslr.org/53/. License: CC BY-SA 4.0.

This is the corpus. Archives 0–7 are the training audio. Archive f is the 100-clip test, CER 0.165. The paper releases the data. It does not report a 16 million parameter recognizer.

### Conneau et al., 2022

A. Conneau et al., “FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech,” arXiv:2205.12446, 2022. https://arxiv.org/abs/2205.12446

FLEURS `bn_in` test, 920 clips, is the first external test. We did not train on FLEURS. Our CER there is 0.659. Their paper evaluates massively multilingual models, not this 16 million parameter student.

### Ardila et al., 2020

R. Ardila et al., “Common Voice: A Massively-Multilingual Speech Corpus,” *LREC*, 2020. https://aclanthology.org/2020.lrec-1.520/

Common Voice 17 Bengali test, 9,327 clips, is the second external test. Our CER is 0.538. The 2020 paper introduces the corpus. It is not a Bangla system paper, and it is not the same split as a private 14.73 WER.

### Radford et al., 2022

A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” arXiv:2212.04356, 2022. https://arxiv.org/abs/2212.04356

Whisper is the family of the baseline. Whisper Tiny is about 39 million parameters. The baseline in our table is a Bangla fine-tune of that size, not the original English-centric checkpoint. Whisper Small is about 244 million parameters. Whisper Large-v2 is about 1.5 billion. None of those sizes is 16 million.

### IJACSA survey, 2021

S. Sultana, M. S. Rahman, and M. Z. Iqbal, “Recent Advancement in Speech Recognition for Bangla: A Survey,” *IJACSA*, vol. 12, no. 3, 2021. DOI [10.14569/ijacsa.2021.0120365](https://doi.org/10.14569/ijacsa.2021.0120365).

The survey reviews Bangla recognizers and public corpora, including OpenSLR 53, and treats Bangla training data as limited. That is a 2021 description of the data. It is not evidence that no recognizer existed, and it is not a 16 million parameter result.

### arXiv:2507.01931, 2025

“Whisper Turns Stronger at Transcribing Bengali Speech,” arXiv:2507.01931, 2025. https://arxiv.org/abs/2507.01931

The paper compares Whisper Small and Whisper Large-v2 with Wav2Vec-BERT on Common Voice 17 and OpenSLR. These are much larger models. The metric they print is word error rate. It is not our character error, and the OpenSLR split is not identified as our 100 clips from archive f. Cite the paper for the large-model comparison. Do not place their WER beside 0.165.

### shhossain/whisper-tiny-bn

Hugging Face model card, not a journal article. https://huggingface.co/shhossain/whisper-tiny-bn

37,760,640 parameters. This is the baseline in our table. Same three tests, same scorer: OpenSLR CER 0.279, FLEURS CER 0.408, Common Voice CER 0.331. This is the one public system whose numbers can sit next to ours. It is larger (about 39 million parameters, not 16).

### thesven/whisper-tiny-bn-v2

Hugging Face model card. https://huggingface.co/thesven/whisper-tiny-bn-v2

A Whisper Tiny fine-tune, about 39 million parameters, trained on OOD-Speech (about 10,000 samples). Different test set. Useful as a second example of a lightweight Bangla fine-tune. Its score is not comparable to 0.165.

### MediBeng, 2025

medRxiv preprint, DOI [10.1101/2025.04.25.25326406](https://doi.org/10.1101/2025.04.25.25326406). Whisper Tiny on synthetic clinical Bengali–English code-switch. The card reports word error 0.01. That test is not OpenSLR, not FLEURS, and not Common Voice. A preprint is not a journal article. Do not paste 0.01 next to 0.165.

### hassanaliemon/BanglaASR

Hugging Face model card. https://huggingface.co/hassanaliemon/BanglaASR

Whisper Small, about 244 million parameters. The card reports 14.73 word error on the authors’ own split of about 23.8 hours. Different size, different metric, different split.

### SaniyatMushrat/SusathoASR

Hugging Face model card. https://huggingface.co/SaniyatMushrat/SusathoASR

XLS-R 300 million parameters, trained on OpenSLR 53. The card reports CER 0.072 and WER 0.311 on a 10% split of 21,871 examples, with no language model. That split is not our 100 clips from archive f. The model is about twenty times larger. Mention the shared corpus and the different split. Do not treat 0.072 as a better score on our test.

### SayedShaun FastConformer

Public model notes for a large FastConformer hybrid trained on Common Voice, OpenSLR 53, and FLEURS `bn_in` training audio. Reported 3.96% character error on the authors’ held-out set. They trained on FLEURS. We did not. Their held-out set is not our 100 clips, and the model is not a 16 million parameter student. Do not paste 3.96% beside 0.165.

### Das et al., 2025, Sylheti

R. K. Das and colleagues, Sylheti speech recognition, *IJISC* 12(02), 2025, DOI [10.30954/2348-7437.2.2025.6](https://doi.org/10.30954/2348-7437.2.2025.6).

Sylheti is a nearby language. It is not standard Bangla. One sentence in related work is enough. Do not use its number as a Bangla result.

## What can share a table with 0.165

| Source | Year | Size | Data | Reported number | Same test as 0.165? |
|:--|--:|--:|:--|:--|:--|
| This checkpoint, rslt3 | 2026 | 16.0M | OpenSLR 53, archives 0–7 | CER 0.165 on archive f, 100 clips | yes |
| shhossain/whisper-tiny-bn | model card | 37.8M | Bangla fine-tune | CER 0.279 on the same 100 clips | yes |
| shhossain/whisper-tiny-bn | model card | 37.8M | same three tests | FLEURS CER 0.408; Common Voice CER 0.331 | same external tests, not the 100 clips |
| Kjartansson et al. | 2018 | data paper | OpenSLR 53 | corpus release | the data, not a score |
| SusathoASR | model card | 300M | OpenSLR 53, their 10% split | CER 0.072 | no |
| SayedShaun FastConformer | model notes | large | OpenSLR + Common Voice + FLEURS train | 3.96% CER, their held-out set | no |
| arXiv:2507.01931 | 2025 | Whisper Small / Large-v2 | Common Voice 17 and OpenSLR | WER, not CER | no |
| hassanaliemon/BanglaASR | model card | ~244M | their own split | 14.73 WER | no |
| MediBeng | 2025 preprint | Whisper Tiny | synthetic clinical code-switch | 0.01 WER | no |
| thesven/whisper-tiny-bn-v2 | model card | ~39M | OOD-Speech | different test | no |
| Das et al. | 2025 | Sylheti system | Sylheti | not standard Bangla | no |

## Sentences you can paste into Section 2

OpenSLR 53 (Kjartansson et al., SLTU 2018) is the training corpus and the source of the 100-clip test. FLEURS (Conneau et al., 2022) and Common Voice (Ardila et al., LREC 2020) are the external tests. The student is not a Whisper checkpoint. The baseline is a public Whisper Tiny Bangla fine-tune with 37.8 million parameters (shhossain/whisper-tiny-bn). Whisper itself is the weak-supervision model of Radford et al. (2022). Sultana, Rahman, and Iqbal (IJACSA, 2021, DOI 10.14569/ijacsa.2021.0120365) surveyed Bangla recognizers and public corpora, including OpenSLR 53, and described the training data as limited. Later work compares much larger models: Whisper Small and Large-v2 against Wav2Vec-BERT (arXiv:2507.01931). Public lightweight Bangla systems that we could find are Whisper Tiny fine-tunes near 39 million parameters, including whisper-tiny-bn and whisper-tiny-bn-v2. We did not find a peer-reviewed paper for a custom Bangla recognizer at about 16 million parameters.

## Do not paste

- MediBeng’s 0.01 word error. It is a preprint on synthetic clinical code-switch, not one of our three tests.
- The FastConformer 3.96% character error. That system trained on FLEURS and scored a different held-out set.
- SusathoASR’s CER 0.072. That is a 300 million parameter model on a 10% OpenSLR split, not the 100 clips in archive f.
- Any sentence that says a bigger GPU would lower 0.165. The saved `student.pt` prints the same text on a faster machine. A new training run would be a different checkpoint.
- Any sentence that says no Bangla speech research exists. It does. The missing piece is a peer-reviewed custom model near 16 million parameters, not the whole field.
