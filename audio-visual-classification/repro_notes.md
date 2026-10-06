# Reproduction Notes

## 2026-10-04 — Standalone controlled-noise evaluation

- Added `test.py` instead of reusing the training entry's hard-coded evaluation branch.
- Reused `CramedDataset` and `KSDataset_Noise` for split selection, sample loading, preprocessing, frame selection, and labels.
- Added a thin evaluation wrapper that applies independently configured visual and audio corruption after the repository dataset returns a clean test sample.
- Gaussian corruption reuses `AddGaussianNoise` and `AddGaussianNoise_spec`. Their existing convention is preserved: strength `0` or `1` produces a clean input.
- Visual salt-and-pepper corruption reuses `AddSaltPepperNoise`; its strength is a density in `[0, 1]`. Since the repository has no audio salt transform, audio salt-and-pepper uses the same density and replaces selected spectrogram values with that sample's minimum or maximum.
- Evaluation reads fused, audio-only, and visual-only logits from one model forward pass and reports all three accuracies.
- Test loading uses `shuffle=False` and `drop_last=False`; this includes all 744 CREMAD test samples instead of the 704 samples evaluated by the old training entry at batch size 64.
- Added explicit controls for dataset, checkpoint, corruption, data paths, frame count, fusion, probabilistic embedding, GPU/device, batch size, workers, seed, optional smoke-test sample limit, and JSON output.
- Added a local compatibility alias for the deprecated `np.float` used inside `KSDataset_Noise`; the training dataset source remains unchanged.
- Updated the README evaluation section with clean, Gaussian, and salt-and-pepper commands.

Validation status:

- Import/help check passed in the `torch2.5.1` environment (actual PyTorch 2.7.0 + CUDA 12.8).
- CREMAD clean checkpoint smoke test passed on 4 samples with CUDA and returned fused/audio/visual accuracies.
- CREMAD Gaussian smoke test passed on 2 samples at visual strength 4 and audio strength 2.
- CREMAD salt-and-pepper smoke test passed on 2 samples at visual density 0.1 and audio density 0.05.
- KineticSound dataset smoke test passed on one sample: audio shape `(129, 626)`, visual shape `(3, 3, 224, 224)`.
- Full test-set results: not run as part of this implementation change.

Paper comparison:

- No paper metric or official evaluation configuration was supplied for this change. Any difference from a paper result may come from the selected checkpoint, corruption strength/type, environment version, frame sampling, or the corrected full-test-set sample count.

## 2026-10-04 — QMF-compatible image noise for both modalities

- Audited the official QMF `text-image-classification/src/data/dataset.py`, `helpers.py`, and `train_qmf.py` implementations.
- Added QMF's `--noise_level` / `--noise_type` interface while keeping `--visual_variance` and `--audio_variance` as modality-specific level overrides for compatibility with the requested AV interface.
- Matched QMF Gaussian semantics: outer application probability `0.5`, Gaussian standard deviation `noise_level * 10` in an 8-bit image domain, shared noise across visual RGB channels, and the reference implementation's upper-bound-only clipping before uint8 conversion.
- Matched QMF Salt semantics: outer application probability `0.5`, fixed pixel density `0.1`, and inner execution probability `noise_level / 100`.
- Applied the same image-domain corruption to visual frames and audio spectrograms.
- Added per-sample audio amplitude adaptation: map the spectrogram's actual minimum and maximum to `[0, 255]`, apply QMF image corruption, then map back to the original range. This supports raw amplitudes near `1e-7` without letting a fixed pixel-scale perturbation overwhelm the signal.
- Verified that the current CREMAD reader returns a log spectrogram with observed sample range `[-16.1174, 1.4918]`; the adaptive mapping therefore uses the observed tensor range rather than assuming the post-reader values remain at `1e-7`.

Validation status:

- Updated help/import check passed in the `torch2.5.1` environment.
- QMF Gaussian checkpoint smoke test passed on 2 CREMAD samples at shared level 5.
- QMF Salt checkpoint smoke test passed on 2 CREMAD samples at shared level 100, which guarantees the inner Salt transform whenever the outer `0.5` gate executes.
- Full test-set results: not run as part of this implementation change.

## 2026-10-04 — Move audio corruption directly behind STFT

- Corrected the audio insertion point following user feedback. The previous implementation corrupted the log spectrogram returned by the base dataset.
- `CramedDataset` and `KSDataset_Noise` now accept an optional `args.audio_noise_transform` callback in test mode.
- The callback is executed in the dataset as `librosa.stft -> abs -> audio noise -> log(+1e-7)`.
- `test.py` installs `AdaptiveSpectrogramNoise` as that callback and no longer corrupts audio after `Dataset.__getitem__` returns.
- The callback maps the raw STFT magnitude range to `[0, 255]`, applies the QMF image transform, maps back to the original nonnegative magnitude range, and only then allows the dataset to take the logarithm.
- Training behavior is unchanged because training launchers do not provide `audio_noise_transform`, and the callback is only called in test mode.

Validation status:

- Syntax/import check passed.
- The callback observed raw CREMAD STFT magnitude shape `(257, 188)` and range `[6.8239e-11, 4.4451]` before logarithmic compression.
- Clean-path identity check passed exactly: `array_equal=True`, maximum absolute difference `0.0`.
- Post-STFT Gaussian checkpoint smoke test passed on 2 CREMAD samples at level 5.
- Post-STFT Salt checkpoint smoke test passed on 2 CREMAD samples at level 100.
- Gaussian smoke test also passed with `num_workers=2`, confirming the callback works in worker processes.
- KineticSound post-STFT Gaussian dataset smoke test passed with audio shape `(129, 626)` and visual shape `(3, 3, 224, 224)`.

## 2026-10-04 — Full seven-model noise matrix

- Added `run_noise_matrix.py` to discover every checkpoint-bearing directory below `results/cramed`, select the checkpoint with the highest accuracy encoded in its filename, and run each requested noise condition independently.
- Discovered seven current model versions: `udml`, `udml_clean`, `udml_clean__fc_2`, `udml_clean__fc_varloss_1`, `udml_clean_no_fc`, `udml_noise__fc`, and `udml_noise__fc_cyccle70`.
- Locked the comparison to CREMA-D test, seed 0, batch size 64, 8 workers, `drop_last=False`, and the four conditions Gaussian 5/10 and Salt 5/10.
- Audio and visual corruption use separate random draws, so each modality independently passes through the outer probability-0.5 gate.
- The runner writes one JSON per model/condition, updates `noise_matrix.csv` after every completed run, and reuses existing JSON files on restart.
- Full command:
  `conda run --no-capture-output -n torch2.5.1 python run_noise_matrix.py --output-dir results/noise_matrix_seed0_p05 --batch-size 64 --num-workers 8`

Validation status:

- All seven selected checkpoints first passed a 2-sample Gaussian-5 CUDA smoke test.
- The full matrix completed all 28 evaluations without retries or OOM errors.
- Every evaluation contains exactly 744 samples; all 28 JSON result files exist; `noise_matrix.csv` contains 28 data rows.
- Best fused accuracy by condition: Gaussian 5 `udml_clean__fc_varloss_1` 40.59%; Gaussian 10 `udml_noise__fc_cyccle70` 37.90%; Salt 5 `udml_clean_no_fc` 74.46%; Salt 10 `udml_clean__fc_varloss_1` 71.91%.
- Highest four-condition mean fused accuracy: `udml_clean_no_fc`, 56.08%.
- Detailed outputs and interpretation are in `results/noise_matrix_seed0_p05/report.md` and `results/noise_matrix_seed0_p05/noise_matrix.csv`.

Paper/reference comparison:

- These results are not directly comparable with the QMF paper because QMF evaluates text-image data, while this project evaluates visual frames plus an adaptively rescaled audio STFT magnitude.
- The evaluated checkpoints are local project variants with different training histories and architectures.
- The original training evaluator used `drop_last=True` and scored 704 samples at batch size 64; this matrix scores all 744 test samples.
- Salt level 5/10 follows the QMF helper's nested probabilities: outer probability 0.5 and inner probability 0.05/0.10, so the effective execution probability per modality is 2.5%/5%. Gaussian applies level-scaled noise whenever its outer gate fires.
- The matrix uses one fixed stochastic seed. Differences below roughly one percentage point require multi-seed evaluation before a paper-level claim.

## 2026-10-04 — Move all noise decisions into Dataset `__getitem__`

- Removed `ControlledNoiseDataset`, `AdaptiveSpectrogramNoise`, and the hidden `args.audio_noise_transform` callback.
- Added four direct QMF-aligned functions to both active Dataset files: `add_gaussian_visual`, `add_salt_visual`, `add_gaussian_audio`, and `add_salt_audio`.
- Visual noise is inserted directly into the per-sample torchvision `Compose`; audio noise is called directly after `abs(librosa.stft(...))` and before `log(... + 1e-7)`.
- Removed initialization-time `a_variance_list` and `v_variance_list`. Training variances are now generated on every `__getitem__` call.
- Training audio and visual each use an independent probability gate (default 0.5) and an inclusive integer range (default 1--11). Test uses fixed modality-specific variances and independently configurable probabilities.
- Salt keeps the QMF nested behavior: the modality gate runs first, then the level-dependent `variance / 100` gate; an applied transform changes 10% of pixels.
- Any modality whose noise is not actually executed returns variance 1 to the model loss.
- Kept the existing `cylcle_epoch` behavior: the clean Dataset is selected before the boundary and the random-noise Dataset afterwards.

Validation status:

- Syntax checks and CLI help passed in the `torch2.5.1` environment.
- Synthetic audio magnitudes spanning `1e-10` to `1e-7` were changed successfully by both Gaussian and Salt adaptive functions without NaN/Inf values.
- With probability 0, the noisy Dataset path was exactly equal to the clean path and returned `(visual_variance, audio_variance) = (1, 1)`.
- With probability 1, fixed Gaussian and Salt levels were applied to both modalities; audio-only probability settings affected audio independently.
- Eight training samples with forced noise produced different sample-level variance pairs inside the configured visual 2--3 and audio 4--5 ranges.
- CREMA-D and KineticSound Dataset smoke checks passed; KineticSound returned audio `(129, 626)` and visual `(3, 3, 224, 224)`.
- A real checkpoint completed forward and backward passes for one clean batch and one post-`cylcle_epoch` noise batch. The clean batch returned all ones; the noise batch returned visual `[2, 3]` and audio `[5, 4]`.
- All seven checkpoints passed a two-sample Gaussian-5 inference check through the new direct Dataset path.
- The full 28-run matrix completed successfully with 744 samples per run and 28 JSON files.

Full-matrix summary (fused accuracy):

- `udml`: Gaussian 5/10 = 37.37/36.83%; Salt 5/10 = 70.16/67.88%; mean 53.06%.
- `udml_clean`: 37.63/32.53%; 71.51/70.03%; mean 52.92%.
- `udml_clean__fc_2`: 30.65/34.01%; 68.95/67.61%; mean 50.30%.
- `udml_clean__fc_varloss_1`: 37.37/29.84%; 73.39/70.97%; mean 52.89%.
- `udml_clean_no_fc`: 38.84/35.22%; 72.58/72.18%; mean 54.70%.
- `udml_noise__fc`: 37.37/36.02%; 69.76/68.68%; mean 52.96%.
- `udml_noise__fc_cyccle70`: 38.84/38.17%; 69.89/66.94%; mean 53.46%.
- Detailed outputs are in `results/noise_matrix_getitem_seed0_p05/`; `comparison_with_wrapper.csv` records every old/new metric delta.

Difference from the previous wrapper run:

- Core QMF formulas and sample count are unchanged, but the sample-level probability decisions now occur inside DataLoader workers with Python RNG instead of the wrapper/callback's PyTorch RNG.
- Visual noise now operates directly on the PIL image inside torchvision `Compose` instead of inverting a normalized tensor in an outer wrapper.
- Those intentional changes alter the exact stochastic subset and produce fused deltas from -3.23 to +1.21 percentage points across individual conditions.
- This remains different from the QMF paper because the modalities, checkpoints, model, dataset split, and audio magnitude adaptation are specific to this project.

## 2026-10-04 — Audit test defaults against official UDML

- Audited the pristine repository `HEAD` (`9fbaca3`) rather than the modified working tree. The released audio-visual Dataset samples training levels as integers 1--11, but its original test branch is effectively clean: CREMA-D hard-codes level 0 and KineticSound hard-codes the identity level 1.
- Audited the UDML paper evaluation protocol. Table 3 corrupts 50% of samples with Gaussian or Salt noise and reports strengths 5 and 10; Gaussian 5 is the first noisy condition.
- Changed standalone `test.py` defaults to `--noise_type Gaussian`, `--noise_level 5`, `--visual_noise_prob 0.5`, and `--audio_noise_prob 0.5`. Explicit modality levels still override the shared level, and `--noise_type None` now reports all effective levels as 0.
- The inference path matches the official evaluator: test split, `spec.unsqueeze(1).float()`, image tensor, `AVClassifier_AUXI_UDML`, and fused/audio/visual logits at output positions 2/9/10. The standalone evaluator keeps all samples (`drop_last=False`) while the released training entry used `drop_last=True`.
- The current corruption implementation intentionally does not reproduce the released Dataset helper byte-for-byte. The release adds visual Gaussian noise with NumPy `scale=variance**2`, adds audio Gaussian noise after logarithmic compression, and does not expose a noisy test CLI. The current code retains the requested QMF semantics and applies adaptively rescaled audio corruption after STFT magnitude and before log.

Validation status:

- `python -m py_compile test.py` passed in the remote `torch2.5.1` environment.
- The RTX 5090 was idle before inference (0 MiB reported used, 0% utilization).
- A real two-sample CUDA run with no noise arguments completed and reported `noise_type=Gaussian`, shared/audio/visual level 5, and both probabilities 0.5.

## 2026-10-05 — Full retest with unimodal results

- Re-ran all seven checkpoint families on the complete 744-sample CREMA-D test split under Gaussian 5/10 and Salt 5/10.
- Kept the controlled settings unchanged: corrected direct Dataset noise path, seed 0, batch size 64, 8 workers, and independent audio/visual corruption probability 0.5.
- All 28 evaluations completed successfully. Every result JSON contains fused, audio-only, and visual-only accuracy; `noise_matrix.csv` contains 28 data rows.
- The retest is exactly reproducible against `noise_matrix_getitem_seed0_p05`: the maximum absolute delta across all 84 fused/audio/visual metric cells is 0.
- Best four-condition mean fused accuracy: `udml_clean_no_fc`, 54.70%.
- Best four-condition mean audio-only accuracy: `udml_clean`, 48.76%.
- Best four-condition mean visual-only accuracy: `udml`, 57.09%.
- Full tables and per-condition winners are in `results/noise_matrix_retest_20261005_seed0_p05/report.md`.

## 2026-10-05 — Add explicit noise launch scripts

- Added `test_noise.sh` for one checkpoint and one condition. Its positional arguments expose dataset, checkpoint, noise type, visual variance, audio variance, GPU, and optional output JSON directly.
- Added `run_noise_matrix.sh` for all discovered checkpoint families under Gaussian 5/10 and Salt 5/10.
- Both launchers keep the controlled defaults visible: independent modality probability 0.5, seed 0, batch size 64, and 8 workers.

## 2026-10-06 — Use QMF zero as the clean variance label

- Changed the default/no-op `visual_variance` and `audio_variance` returned by both active Datasets from 1 to 0.
- Changed the default training minimum for both modalities from 1 to 0. The configured maximum remains 11.
- Changed Dataset fallback minimums and fixed test fallbacks to 0, so a modality whose probability gate does not execute is supervised as clean level 0.
- Kept the probabilistic embedding KL mathematically valid by mapping a clean target label 0 to the original unit-variance prior only inside `regurize`; the variance estimator MSE still receives the requested target 0.
- Probability values 0 and 1 now resolve directly without consuming a random draw. This keeps the KineticSound random audio crop identical between `add_noise=False` and an `add_noise=True, probability=0` check.

Validation status:

- Syntax compilation passed for both Dataset files, the training entry, and `test.py`.
- CREMA-D and KineticSound probability-0 paths exactly matched their clean inputs and returned `(visual_variance, audio_variance) = (0, 0)`.
- A CREMA-D training sample with both ranges forced to `[0, 0]` returned `(0, 0)`.
- The zero-label KL compatibility path completed finite CUDA forward and backward passes.

## 2026-10-06 — Standard-normal KL with std-based FC uncertainty

- Replaced the noise-level-dependent Gaussian KL with the standard `KL(N(mu, std^2) || N(0, I))`; noise labels no longer enter the KL term.
- Kept uncertainty estimation on the requested path: pooled `a_std` / `v_std`, detached from the encoder, are passed to the existing modality-specific FC estimators.
- Replaced `exp(0.5 * output) + 1` with `softplus(output)`, allowing the scalar FC prediction to approach the QMF clean label 0 while remaining nonnegative.
- Added `1e-8` to both uncertainty-weight normalization denominators. No Dataset API, model parameter shape, return position, or extra configuration mode was added.

Validation status:

- Syntax compilation passed for the training entry, model file, and test entry.
- Standard-normal KL returned zero for `mu=0, std=1` and completed finite backward propagation.
- A focused gradient check confirmed that the noise MSE updates the FC estimator while `.detach()` blocks that loss from the std input.
- A real CREMA-D checkpoint loaded strictly under the unchanged parameter shapes.
- One clean batch and one Gaussian-noise batch each completed forward and backward propagation with finite KL, FC loss, total loss, uncertainty predictions, and fusion weights.
- The clean batch returned visual/audio labels `[0, 0]`; the noise batch returned visual `[3, 2]` and audio `[5, 5]` within the configured ranges.
- Existing checkpoints were trained with the previous KL and output transform; their historical accuracy remains an old-definition result and is not treated as a result of this new formulation.

## 2026-10-06 — Add cycle-noise training launcher

- Added `train_udml_noise_cycle.sh` for CREMA-D UDML training with a clean stage before `cylcle_epoch` and random-noise training afterwards.
- Defaults are Gaussian, cycle 50, GPU 0, 100 epochs, batch size 64, audio/visual ranges 0--11, and independent modality probability 0.5.
- Noise type, cycle boundary, GPU, checkpoint directory, ranges, probabilities, epochs, and batch size remain directly configurable from arguments or environment variables.

## 2026-10-06 — Evaluate cycle-50 Gaussian checkpoint

- Evaluated `udml_noise_cycle50_gaussian_0_11` epoch 71 (`acc_0.7372159090909091`) on all 744 CREMA-D test samples.
- Used seed 0 and independent audio/visual corruption probability 0.5 for Gaussian 5 and 10.
- Clean fused/audio/visual accuracy: 72.72/58.33/63.98%.
- Gaussian 5 fused/audio/visual accuracy: 68.28/49.46/57.26%.
- Gaussian 10 fused/audio/visual accuracy: 65.19/47.98/49.73%.
- Fixed `test_noise.sh` to pass the shared `--noise_level` when audio and visual strengths are equal, so compatibility metadata now agrees with the effective modality-specific levels.
- Raw JSON and a compact report are in `results/eval/udml_noise_cycle50_gaussian_0_11_epoch71_20261006/`.
- Salt 5 fused/audio/visual accuracy: 71.24/57.26/65.46%.
- Salt 10 fused/audio/visual accuracy: 70.70/56.45/63.44%.
- Salt uses the QMF nested probability, giving effective per-modality transform probabilities of 2.5% and 5% for levels 5 and 10.

## 2026-10-06 — Change Salt corrupted-element density to 50%

- Changed visual and audio Salt masks in both CREMA-D and KineticSound from 10% to 50% corrupted elements: 25% pepper, 25% salt, and 50% unchanged.
- Kept the independent outer modality probability and the inner `level / 100` execution gate unchanged. With the default outer probability 0.5, Salt 5 and Salt 10 still execute for 2.5% and 5% of samples per modality.
- This requested 50% density is deliberately different from the QMF reference density of 10%; only the nested gating and level semantics remain aligned with QMF.

Validation status:

- Syntax compilation passed for both Dataset files and `test.py`.
- A 1,000,000-element statistical check measured visual pepper/salt/unchanged ratios of 25.06/24.97/49.98% and audio ratios of 25.26/24.97/49.78% in both Dataset modules.
- One real Salt sample from each Dataset completed successfully: CREMA-D returned spectrogram `(257, 188)` and images `(3, 1, 224, 224)`; KineticSound returned `(129, 626)` and `(3, 3, 224, 224)`. Both returned finite values and noise labels `(100, 100)` under guaranteed execution.
- Re-evaluated the cycle-50 Gaussian checkpoint on all 744 CREMA-D test samples. At 50% Salt density, Salt 5 fused/audio/visual accuracy is 70.56/56.85/64.78%; Salt 10 is 68.41/55.65/61.42%.
- Relative to the previous 10% density results, Salt 5 changes by -0.67/-0.40/-0.67 percentage points and Salt 10 by -2.28/-0.81/-2.02 points for fused/audio/visual accuracy.
- Raw JSON and the comparison table are in `results/eval/udml_noise_cycle50_gaussian_0_11_epoch71_salt_density50_20261006/`.

## 2026-10-06 — Remove the Salt inner probability gate

- Removed the Salt-specific `level / 100` probability check from CREMA-D and KineticSound.
- Salt now uses only the independent outer audio/visual probability, which defaults to 0.5. A selected modality always receives the 50% corrupted-element mask.
- Salt levels 5 and 10 now produce the same corruption operation and remain different only as variance labels returned to the model.
- This experiment setting no longer follows QMF Salt gating or pixel density.

Validation status:

- Syntax compilation passed for both Dataset files and `test.py`, and a source check confirmed that no `level / 100` Salt gate remains.
- With outer probabilities fixed to 1 and Salt level 5, one real sample from each Dataset returned `(visual_variance, audio_variance) = (5, 5)` and finite tensors. Under the removed gate and seed 0, the previous implementation would have returned clean labels.
- Re-evaluated the cycle-50 Gaussian checkpoint on all 744 CREMA-D test samples with seed 0 and outer audio/visual probability 0.5.
- Salt 5 and Salt 10 both produced fused/audio/visual accuracy of 40.73/36.96/41.80%. They are identical because the Salt pixel operation no longer depends on level and inference does not consume the returned variance label.
- Compared with the preceding 50%-density plus inner-gate results, Salt 5 changed by -29.84/-19.89/-22.98 percentage points and Salt 10 by -27.69/-18.68/-19.62 points for fused/audio/visual accuracy.
- Raw JSON and the report are in `results/eval/udml_noise_cycle50_gaussian_0_11_epoch71_salt_outer_only_density50_20261006/`.

## 2026-10-06 — Set outer-only Salt density to 10%

- Changed visual and audio Salt masks in CREMA-D and KineticSound to 10% corrupted elements: 5% pepper, 5% salt, and 90% unchanged.
- Kept the Salt inner probability gate removed. Only the independent outer modality probability, default 0.5, determines whether the fixed-density mask is applied.
- Salt levels 5 and 10 continue to use the same corruption operation and differ only as returned variance labels.

Validation status:

- Syntax compilation passed for both Dataset files and `test.py`.
- A 1,000,000-element statistical check measured visual pepper/salt/unchanged ratios of 5.00/5.02/89.98% and audio min/max/other ratios of 5.35/5.02/89.63% in both Dataset modules. The small audio-min excess comes from unchanged quantized values already at the source minimum.
- One real Salt level-5 sample from each Dataset completed with finite tensors and returned `(visual_variance, audio_variance) = (5, 5)` under outer probability 1, confirming that the inner gate remains absent.
- Re-evaluated the cycle-50 Gaussian checkpoint on all 744 CREMA-D test samples with seed 0 and outer audio/visual probability 0.5.
- Salt 5 and Salt 10 both produced fused/audio/visual accuracy of 51.48/43.01/55.78%. They remain identical because level does not alter the Salt operation.
- Relative to outer-only Salt at 50% density, reducing density to 10% changed fused/audio/visual accuracy by +10.75/+6.05/+13.98 percentage points.
- Raw JSON and the report are in `results/eval/udml_noise_cycle50_gaussian_0_11_epoch71_salt_outer_only_density10_20261006/`.

## 2026-10-06 — Restore the QMF Salt definition

- Restored the Salt-specific `level / 100` inner probability gate in CREMA-D and KineticSound.
- Kept the QMF corrupted-element density at 10%: 5% pepper, 5% salt, and 90% unchanged.
- Salt now again requires both the independent outer modality gate and the inner level gate. With outer probability 0.5, effective per-modality execution probabilities are 2.5% for level 5 and 5% for level 10.

Validation status:

- Syntax compilation passed for both Dataset files and `test.py`.
- The 10% mask check measured visual pepper/salt/unchanged ratios of 5.00/5.02/89.98% and audio min/max/other ratios of 5.35/5.02/89.63% in both Dataset modules.
- A controlled real-sample test forced the inner random value to 0 and 0.99 at Salt level 5. Both CREMA-D and KineticSound returned `(5, 5)` for the passing case and `(0, 0)` for the failing case.
- Re-evaluated the cycle-50 Gaussian checkpoint on all 744 CREMA-D test samples with seed 0 and outer audio/visual probability 0.5.
- Salt 5 fused/audio/visual accuracy returned to 71.24/57.26/65.46%; Salt 10 returned to 70.70/56.45/63.44%.
- Both restored result JSON files are byte-for-byte identical to the earlier QMF-definition result files.
- Raw JSON and the report are in `results/eval/udml_noise_cycle50_gaussian_0_11_epoch71_salt_qmf_restored_20261006/`.

## 2026-10-06 — Refresh CREMA-D and KineticSound training launchers

- Replaced the duplicate CREMA-D commands and the stale KineticSound absolute output path with one canonical launcher per dataset.
- Both launchers now default to explicit clean training (`noise_type=None`) and accept positional `NOISE_TYPE CYCLE_EPOCH GPU_ID [CKPT_DIR]` arguments for clean, Gaussian-cycle, or Salt-cycle runs.
- Made the current 0--11 training ranges, independent 0.5 modality probabilities, clean validation settings, initial depend values 32/10, optimizer, learning rate, and dataset-specific UDML hyperparameters explicit.
- Added environment overrides for Python, epochs, batch size, seed, variance ranges, noise probabilities, and initial depend values.

Validation status:

- `bash -n` passed for both launchers.
- Dry-run command expansion passed for clean, Gaussian, and Salt configurations across both datasets.
- Zero-epoch real initialization completed for CREMA-D and KineticSound without entering a training batch.
- Normalized both launcher files to Unix LF endings after the first smoke test exposed a trailing carriage return in the final `gpu_ids` argument.

## 2026-10-06 — Simplify the dataset training launchers

- Replaced launcher arguments, environment overrides, and validation branches with two fixed commands per dataset.
- Each script now runs clean UDML first and Gaussian cycle-50 training second.
- Removed explicit depend, variance-range, and noise-probability arguments; these use the training parser defaults.
- Kept only dataset-specific output paths, frame counts, beta, and gamma plus the flags needed to distinguish clean from Gaussian training.
