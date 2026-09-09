# Wire laser camera project

Snapshot of the current Python programs in D:\project\scripts.
Existing Python source contents are preserved without modification.

## Runtime layout and external dependencies

The current config.py expects PROJECT_ROOT = D:\project. Keep this checkout
at D:\project\scripts to retain the existing paths without code changes.

The following inputs live outside this repository and are not included:

- D:\project\calibration: camera, metric-plane, cone and angle calibration resources.
- D:\project\runs: acquisition data and generated experimental results.

A checkout alone is therefore not a complete runnable experimental dataset.
Install the Python libraries and hardware SDKs required by the individual
programs and provide the external inputs before running acquisition or analysis.
No dependency versions have been guessed or pinned for this snapshot.

## Exclusions

The 16 existing __pycache__ files are generated Python caches.
The root yolo11n.pt (5,613,764 bytes) is not referenced by the current Python
sources; config.py uses the trained best.pt above. It is retained locally.
Temporary files, logs, archives, editor settings, virtual environments and
common credential files are ignored. No blanket image, JSON, YAML, NPZ or
model extension exclusions are applied. No Git LFS is used.

## Included trained model

models/segment/train_mixed_v3/weights/best.pt is a byte-identical backup of
D:\project\model\segment\train_mixed_v3\weights\best.pt (6,750,637 bytes).
On a new machine, copy this file to that external path before running
segmentation: config.py is unchanged and still reads that external path.
The original model remains in place. The weight file is stored directly
in Git without Git LFS. The unused root yolo11n.pt remains excluded.
