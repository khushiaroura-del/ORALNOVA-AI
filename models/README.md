# Models Directory

Trained model checkpoints are saved here.

## Files

- `oralnova_best.pth` — Best validation checkpoint (created during training)
- `oralnova_last.pth` — Last epoch checkpoint

## Training

Run `python -m src.train --data_dir data` to generate checkpoints.

> **No pre-trained weights are bundled with this project.**
> Predictions require real training on a compatible dataset.
