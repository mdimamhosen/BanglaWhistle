# A Compact Syllable-Token Bangla Recognizer: In-Domain Gains and Cross-Corpus Limits

**Anonymous submission**<br>
October 2026

## Abstract

We describe a 15.97-million-parameter Bangla automatic speech recognizer with a 63.94 MB float32 checkpoint. The system uses a six-layer encoder and decoder, 256-dimensional hidden states, and a vocabulary built from Bangla orthographic syllable pieces. It was trained using OpenSLR 53 training archives 0–7 with a frozen Whisper-small encoder as a training guide; the guide is not part of the released recognizer. We evaluate the saved `rslt3` checkpoint with beam width five and Unicode NFC scoring on three test sets. On 100 utterances from OpenSLR 53 archive f, a held-out archive from the same corpus, the model obtains character error rate (CER) 0.165 and word error rate (WER) 0.428, compared with 0.279 and 0.871 for `shhossain/whisper-tiny-bn` (37.76 million parameters). The paired mean utterance-CER gap is −0.133 (95% bootstrap interval −0.174 to −0.091). The direction reverses on FLEURS Bengali (`bn_in`, 920 clips): CER 0.659 vs. 0.408 and WER 0.909 vs. 0.833. It also reverses on the Common Voice 17 Bengali test split (9,327 clips): CER 0.538 vs. 0.331 and WER 0.939 vs. 0.862. These results support a narrow conclusion: this compact model performs better on a held-out split from its training corpus, but the public Whisper baseline generalizes better on the two external test sets. We release an inference package and a small hosted-demo implementation to make the model and its limitations inspectable.

**Keywords:** Bangla, Bengali, speech recognition, low-resource ASR, character error rate, compact models

## 1. Introduction

Automatic speech recognition in Bangla remains difficult to evaluate with a single score. Orthography contains combining signs and conjuncts; spoken varieties, recording conditions, and transcription conventions differ between corpora. A system can perform well on material related to its training data and still fail on a benchmark collected through another process. This paper reports that pattern directly rather than treating model size as a proxy for capability.

We study one saved checkpoint, `rslt3`, a compact encoder–decoder model with 15,968,768 trainable parameters. Its float32 checkpoint occupies 63,944,321 bytes. The model is substantially smaller by parameter count than the public `shhossain/whisper-tiny-bn` checkpoint, which has 37,760,640 parameters. The comparison is not a claim that size alone explains performance: the systems differ in training data, architecture, and decoding procedure.

The central result is split by domain. On OpenSLR 53 archive f, not used in training, `rslt3` has lower CER and WER than the comparison model. On FLEURS Bengali and Common Voice 17 Bengali, the comparison model has lower error on both metrics. The external-set gaps are not small sampling fluctuations under the paired utterance bootstrap recorded in the experiment artifact. We therefore present the model as a compact in-domain result with a clear transfer limitation, not as a generally superior Bangla recognizer.

This report has three practical contributions. First, it documents the saved checkpoint and scoring results using the same values stored in `results.json`. Second, it publishes inference code and a model package that load only the `rslt3` checkpoint. Third, it provides a minimal API and upload page that surface both aggregate results and representative errors. We do not present a tokenizer ablation, claim state-of-the-art performance, or recommend the model for general-purpose transcription without corpus-specific validation.

## 2. Background and Related Work

OpenSLR 53 is a crowd-sourced Bangladeshi Bengali ASR corpus distributed as numbered archives [1]. The training run used archives 0–7; archive f is the held-out evaluation archive used here. This creates an in-domain test in the sense of shared collection and corpus conventions, while preserving a distinct archive for evaluation. It does not establish generalization to speakers or recording conditions outside that corpus.

FLEURS provides parallel speech data across 102 languages and supports ASR evaluation, among other tasks [2]. We use its Bengali `bn_in` test split. Common Voice is a crowd-sourced, multilingual corpus intended for speech technology research [3]. Our Common Voice 17 test transcripts and audio are read from the `fsicoli/common_voice_17_0` mirror recorded in the experiment artifact; the artifact notes that the Mozilla-hosted repository no longer served these audio files at the time the evaluation was assembled. The mirror is identified here to make the provenance explicit rather than implying an official release path.

The comparison system is `shhossain/whisper-tiny-bn`, a Bengali-tuned Whisper Tiny checkpoint [4]. Whisper is an encoder–decoder speech model trained with large-scale weak supervision [5]. For the experiments reported here, the baseline is decoded with a forced Bengali transcription prompt, while `rslt3` uses its own beam search. The two systems are evaluated on the same audio and references within each corpus, but their inference stacks and language-control mechanisms are not identical. The comparison is consequently a practical checkpoint comparison, not a controlled architecture ablation.

## 3. Model and Training Summary

### 3.1 Architecture

The released student is a Transformer encoder–decoder that consumes 80-bin log-Mel features from a Whisper-compatible feature extractor. The encoder and decoder each have six layers, with model width 256, four attention heads, and a feed-forward width of 1,536. A strided convolution reduces the acoustic time axis before the Transformer encoder. The text embedding and output projection are tied. The inference vocabulary has 4,096 slots; the saved Bangla syllable vocabulary contains 1,750 pieces, in addition to reserved token IDs.

The tokenizer groups Bangla graphemes into orthographic-syllable-like pieces: vowel signs and combining marks remain attached to a base, and a hasant joins the following consonant. This description explains how the vocabulary is represented. It is not evidence that syllable pieces outperform characters, subwords, or another tokenizer, because no tokenizer ablation was run.

The parameter count and checkpoint byte count are taken from `results.json`: 15,968,768 parameters and 63,944,321 bytes for `student.pt`. The byte count is 63.94 decimal MB (about 61.0 MiB). The model card for the comparison checkpoint reports 37,760,640 parameters; its hosted safetensors file is approximately 151 MB [4]. Storage sizes depend on the serialized format, so the parameter counts are the more direct measure of model capacity.

### 3.2 Training Data and Procedure

The student was trained on OpenSLR 53 archives 0–7. A frozen Whisper-small encoder supplied guidance during training; those teacher weights are not loaded by the inference code and are not included in the model package. The training run ended when the Kaggle session reached its 12-hour limit. The resulting saved checkpoint, rather than a retrained or post-selected variant, is the sole subject of this report.

The evaluation and service use only `checkpoints/rslt3/student.pt`, its `bn.syllables.json` vocabulary, and its `preprocessor_config.json`. No results from any other local checkpoint are substituted into or combined with the numbers below.

## 4. Evaluation Protocol

### 4.1 Test Sets

We report three test sets. OpenSLR 53 archive f contributes 100 utterances sampled by the evaluation setup from the archive and is in-domain relative to the training corpus. FLEURS `bn_in` test contributes 920 clips and is treated as out-of-domain. The Common Voice 17 Bengali test contributes 9,327 clips and is also treated as out-of-domain. The Common Voice audio source is the `fsicoli/common_voice_17_0` mirror recorded in the results artifact. No FLEURS test clips, Common Voice test clips, or OpenSLR archive-f clips are included among the stated training archives.

These are three finite test collections, not a representative sample of every Bangla speaker, dialect, or use case. “Out-of-domain” here means external to the OpenSLR 53 training corpus; it does not imply a formal domain taxonomy or that the two external corpora cover all deployment conditions.

### 4.2 Decoding and Scoring

The student uses beam search with width five. The Whisper Tiny baseline uses generation with a forced Bengali transcription prompt. Audio is passed through the feature extractor at 16 kHz, and the student is supplied the real mel-frame length rather than the full padded 30-second representation. Output and reference text are normalized to Unicode NFC before scoring. The scoring code rejects baseline hypotheses containing letters outside the Bengali Unicode block, a guard against language-control failure.

CER is Levenshtein character edit distance divided by the number of reference characters after spaces are removed. WER is word edit distance divided by the number of whitespace-delimited reference words. Punctuation and other characters follow the scorer’s stored text; no additional lexical normalization is claimed. Lower values are better. Real-time factor (RTF) is measured compute seconds divided by audio seconds in the saved results. The artifact does not record enough host and runtime detail to treat those RTF values as hardware-independent measurements.

For each test set, the paired interval is computed over the per-utterance CER difference, student minus baseline. The experiment stores the mean utterance-level gap and a 95% percentile interval from 1,000 paired bootstrap resamples with a fixed random seed. This uncertainty interval describes variation across the test utterances under that resampling procedure; it does not account for corpus sampling bias, annotation error, or differences in the decoding stacks.

## 5. Results

Table 1 reports the saved aggregate results. Counts, error rates, and RTF values are copied from `results.json`; parameter counts and the student checkpoint size are recorded there as well. The baseline model size is the hosted safetensors artifact listed in its model repository [4].

| Test set | Clips | System | Parameters (M) | Weights (MB) | CER | WER | RTF |
|:--|--:|:--|--:|--:|--:|--:|--:|
| OpenSLR 53 zip f | 100 | rslt3 | 15.97 | 63.94 | 0.165 | 0.428 | 0.078 |
| OpenSLR 53 zip f | 100 | whisper-tiny-bn | 37.76 | 151 | 0.279 | 0.871 | 0.066 |
| FLEURS bn_in | 920 | rslt3 | 15.97 | 63.94 | 0.659 | 0.909 | 0.273 |
| FLEURS bn_in | 920 | whisper-tiny-bn | 37.76 | 151 | 0.408 | 0.833 | 0.107 |
| Common Voice 17 bn | 9,327 | rslt3 | 15.97 | 63.94 | 0.538 | 0.939 | 0.207 |
| Common Voice 17 bn | 9,327 | whisper-tiny-bn | 37.76 | 151 | 0.331 | 0.862 | 0.106 |

**Table 1.** Aggregate held-out results. OpenSLR archive f is held out from training but belongs to the same corpus; the other two tests are external. RTF is from the recorded experiment environment and should not be read as a portable latency guarantee.

The mean per-utterance CER gap (rslt3 minus baseline) is −0.133 on OpenSLR, with a 95% interval from −0.174 to −0.091. On FLEURS the gap is +0.260 (0.251 to 0.268), and on Common Voice it is +0.184 (0.181 to 0.188). The sign change is consistent with the aggregate CER values: rslt3 has lower error on the related archive and higher error on both external corpora. WER shows the same ordering on all three tests.

Two short examples illustrate the kinds of remaining errors. For the reference **<span class="bn">আমার কপালও</span>**, rslt3 outputs **<span class="bn">আমার কপানো</span>** and the baseline outputs **<span class="bn">আমার কোপালো</span>**. For **<span class="bn">৯২ সালের</span>**, rslt3 outputs **<span class="bn">বেড়ানো বেসালের</span>** and the baseline outputs **<span class="bn">বিড়ানো বই সালের</span>**. Both systems miss both examples; they are failure examples, not a representative qualitative sample.

## 6. Discussion

The main finding is conditional performance. A smaller checkpoint can outperform a larger public model on a held-out archive from the same corpus while underperforming on two external tests. The OpenSLR result is useful evidence that the student learned patterns relevant to that evaluation split. It is not evidence that the model is a better general Bangla recognizer. The external results place a concrete boundary around the in-domain result.

The gap on FLEURS and Common Voice is substantial in both CER and WER. A user choosing a recognizer for those sources should prefer the comparison model based on these measurements, subject to reproducing the same evaluation protocol. The student’s lower parameter count and smaller float checkpoint may matter in constrained settings, but compactness alone does not justify accepting the observed external-set error. A deployment should identify its target corpus and test representative audio before use.

The in-domain CER interval excludes zero under the stored paired bootstrap, and both out-of-domain intervals are positive. That makes the direction of each comparison stable under resampling of the evaluated utterances. It does not repair limitations in the setup. In particular, OpenSLR uses a 100-clip evaluation sample, while the other tests contain more clips and were assembled under different collection and transcription processes. We do not claim that the bootstrap intervals capture variation over speakers, archives, or future recordings.

RTF is below one for each recorded comparison, but it is not a standardized speed study. The saved artifact includes total audio and compute seconds, not a complete report of CPU, thread count, runtime versions, background load, or batching. The model’s public API therefore uses CPU inference as an implementation choice; no latency service-level objective is inferred from Table 1. A single serialized inference path avoids concurrent model calls saturating a small shared machine, while the per-IP limit discourages abusive repeated uploads.

The demo exposes only the three aggregate rows and two named errors. It intentionally does not send the 1.7 MB per-utterance `results.json` to a browser. The benchmark package keeps the test audio and references out of the downloadable model folder. This allows a reader to inspect the model and scorer without turning held-out evaluation material into a training or distribution artifact.

## 7. Limitations and Reproducibility

Training stopped at the Kaggle 12-hour limit. The result is one checkpoint from one run; there is no repeated-seed estimate, training-curve analysis, or hyperparameter sweep in this report. We did not run a SentencePiece-versus-syllable-token ablation, a character-token baseline, or an architecture ablation. Syllable pieces are a tokenizer construction choice, not a demonstrated source of improvement.

The OpenSLR archive-f evaluation contains only 100 clips. The character count in the digit stratum is eight, so digit-stratum CER is too unstable to interpret and is not reported as a finding. The FLEURS and Common Voice results use their existing normalized references; differences in orthographic and annotation conventions may contribute to error, and no manual relabeling study is included. The paired bootstrap resamples utterances, not speakers or data sources.

The comparison is not perfectly controlled. The model architectures, training data, and decoding routines differ. The baseline uses a forced Bengali prompt; the student uses a fixed local vocabulary and beam search. We report the exact saved run rather than claiming a matched architecture comparison. The Common Voice audio was read from the mirror `fsicoli/common_voice_17_0`, as identified in `results.json`, not from a fresh download of an official Mozilla-hosted archive. The artifact does not record a pinned dataset commit, so exact audio revision recovery requires additional provenance beyond the saved JSON.

The inference package is designed for clips no longer than 30 seconds and truncates longer inputs. The web upload accepts files up to 25 MB, transcodes through FFmpeg to 16 kHz mono, and applies a small per-IP request limit. This is a demonstration service, not a privacy-certified or high-availability system. Audio is not retained by application code after the request, but hosting and infrastructure logs are outside the model’s control. The service does not provide a confidence score, speaker attribution, diarization, or a transcript correction interface.

All reported metric values are available in `results.json`; the serving response includes a smaller three-test summary only. The inference image and Hugging Face folder use the `rslt3` checkpoint files only. For the public dataset licenses and baseline model terms, users should review the cited source repositories before redistributing derived artifacts.

## 8. Conclusion

We report a compact 15.97M-parameter Bangla recognizer with lower CER and WER than `whisper-tiny-bn` on a 100-clip held-out OpenSLR archive, and higher CER and WER on FLEURS Bengali and Common Voice 17 Bengali. The evidence supports a corpus-specific advantage and an external-generalization limitation. The released paper, model package, scorer, API, and demo are tied to the same `rslt3` checkpoint and stored results. No claim beyond that comparison is warranted by these experiments.

## References

1. O. Kjartansson, S. Sarin, K. Pipatsrisawat, M. Jansche, and L. Ha. “Crowd-Sourced Speech Corpora for Javanese, Sundanese, Sinhala, Nepali, and Bangladeshi Bengali.” *Proceedings of the 6th International Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU)*, 2018, pp. 52–55. [OpenSLR 53](https://www.openslr.org/53/). DOI: [10.21437/SLTU.2018-11](https://doi.org/10.21437/SLTU.2018-11).
2. A. Conneau et al. “FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.” arXiv:2205.12446, 2022. [arXiv](https://arxiv.org/abs/2205.12446).
3. R. Ardila et al. “Common Voice: A Massively-Multilingual Speech Corpus.” *Proceedings of LREC*, 2020, pp. 4218–4222. [ACL Anthology](https://aclanthology.org/2020.lrec-1.520/).
4. S. Hossain. [whisper-tiny-bn model repository](https://huggingface.co/shhossain/whisper-tiny-bn), accessed October 2026.
5. A. Radford et al. “Robust Speech Recognition via Large-Scale Weak Supervision.” arXiv:2212.04356, 2022. [arXiv](https://arxiv.org/abs/2212.04356).
6. [Common Voice 17.0 Bengali mirror](https://huggingface.co/datasets/fsicoli/common_voice_17_0); the experiment artifact identifies the repository and split but does not include a pinned commit hash.
