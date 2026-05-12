Test Data Notes

This folder stores evaluation datasets for `nvidia_jetson/run_test_inference.py`.

Frame Resize Utility

Use `resize_frames.py` to create resized frame folders (for example, `JPEGImages_256_256` and `JPEGImages_512_512`) from an existing source folder such as `JPEGImages` (uses BILINEAR interpolation).

Mask Resize Utility

Use `resize_masks.py` to create resized mask folders (for example, `SyntheticMasks_256_256` and `SyntheticMasks_512_512`) from an existing source folder such as `SyntheticMasks` (uses NEAREST interpolation to preserve binary mask values).

Examples (run from repo root):

```bash
# Resize frames
python nvidia_jetson/Test_Data/resize_frames.py --data-root nvidia_jetson/Test_Data --datasets DAVIS --src-subdir JPEGImages_Old --dst-prefix JPEGImages --sizes 256x256 512x512

# Resize synthetic masks
python nvidia_jetson/Test_Data/resize_masks.py --data-root nvidia_jetson/Test_Data --datasets DAVIS --src-subdir SyntheticMasks --dst-prefix SyntheticMasks --sizes 256x256 512x512
```

After generation, you can select resized folders via:

```bash
python run_test_inference.py --frames-subdir JPEGImages_256_256 ...
python run_test_inference.py --frames-subdir JPEGImages_512_512 ...
```