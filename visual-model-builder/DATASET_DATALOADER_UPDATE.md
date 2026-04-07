# Dataset and DataLoader Upgrade

This update extends the teaching project from builtin-only datasets to real local image classification data.

## Scope

- Dataset supports `builtin` and `image_folder`.
- `csv` now has schema and inspection placeholders, but runtime/code generation remains intentionally blocked in this phase.
- DataLoader now supports common runtime parameters such as `dropLast`, `pinMemory`, `persistentWorkers`, `prefetchFactor`, and `collateFnType`.

## New Backend API

- `POST /inspect-dataset`
  - Input: Dataset node config
  - Output: `success`, `datasetMode`, `resolvedSplitMode`, `sampleCount`, `numClasses`, `classNames`, `splits`, `inputShape`, `warnings`, `errors`

## Image Folder Support

Supported layouts:

```text
dataset_root/
  train/
    class_a/
    class_b/
  val/
    class_a/
    class_b/
```

or

```text
dataset_root/
  class_a/
  class_b/
```

Behavior:

- Automatically scans class names.
- Counts readable samples.
- Infers `numClasses`.
- Uses predefined train/val/test folders when present.
- Falls back to ratio splitting when only flat class folders exist.

## Frontend Changes

- Dataset and DataLoader node schemas were expanded in the node registry.
- The Inspector now renders dataset-mode-specific fields dynamically.
- Dataset preview calls `/inspect-dataset` and shows:
  - dataset mode
  - split mode
  - sample count
  - class count
  - class names
  - split summary
  - input shape summary
  - warnings and errors

## Runtime and Code Generation

- Training code generation now emits readable dataset helpers for `builtin` and `image_folder`.
- `/run-training` now reuses the same dataset inspection metadata path as validation and code generation.
- Runtime returns dataset summary metadata including dataset mode, class names, split counts, and input shape.

## Validation Coverage

Validation now covers:

- invalid dataset paths
- empty image folders
- invalid split ratios
- empty train split
- class count vs model output mismatch
- input shape vs model input mismatch
- DataLoader parameter legality
- classification/regression loss compatibility warnings
