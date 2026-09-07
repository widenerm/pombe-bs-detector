Google Drive upload instructions

Upload one package folder's contents to a Drive folder such as:

  My Drive/pombe_validation/carmen_package_1/

Each H5 contains a sparse set of annotated frames for one video. The H5
dataset is named "frames" and the original movie frame indices are stored in
"frame_numbers". Select one H5 at a time in the tuning notebook and use the
matching ground-truth.csv.

These sparse H5s are intended for frame-only scar validation. The notebook
uses the annotated scar coordinate to associate each selected cell contour,
and disables tracking/stabilization. They are not suitable for tracking
validation; that later step requires a complete movie H5.

The converter that created these files can process future packages:

  python scripts/convert_validation_tifs_to_h5.py PACKAGE_DIR \
      --output OUTPUT_DIR

It also accepts a ZIP file as PACKAGE_DIR. Do not upload the raw TIFFs or
blinded PNGs for detector tuning; they are retained for manual reference.
