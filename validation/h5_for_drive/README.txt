Sparse H5 files for the development packets (not in git)

dev1/ and dev2/ each hold one H5 per movie with only the annotated frames:
dataset "frames" (the images) and "frame_numbers" (their original movie frame
numbers), plus frame_manifest.csv. validation/scar_cells.py reads them to build
the per-cell records that scar_eval.py, scar_fit.py and scar_interrater.py score:

  python -m validation.scar_cells --cache MASKS --out cells.pkl --segment-missing --precision float32

Rebuild them from an annotation packet with:

  python scripts/convert_validation_tifs_to_h5.py validation/packets/dev1 --output validation/h5_for_drive/dev1
