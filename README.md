# UDML

Unbiased Dynamic Multimodal Fusion for two multimodal classification settings:

- `audio-visual-classification/`: audio-visual classification
- `text-image-classification/`: text-image classification on MVSA-Single

This repository is organized as two independent task folders. Each folder keeps its own scripts, dependencies, and README.

## Repository Layout

```text
UDML/
|- audio-visual-classification/
`- text-image-classification/
```

## How To Use

Choose the task you want, enter that folder, and follow the local README there.

### Audio-visual branch

```bash
cd audio-visual-classification
```

Then read:

```text
audio-visual-classification/README.md
```

### Text-image branch

```bash
cd text-image-classification
```

Then read:

```text
text-image-classification/README.md
```

## Notes

- The two branches are intentionally kept independent.
- Commands are expected to be run from inside the corresponding task folder.
- The text-image branch follows the dataset/model path convention used by QMF.

## Acknowledgement

- QMF layout reference for the text-image branch: <https://github.com/QingyangZhang/QMF>
