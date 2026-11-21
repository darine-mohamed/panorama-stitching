# Panorama Stitching

## Run
    pip install -r requirements.txt
    python run.py data/boat output/boat                 # multi-band blend (default)
    python run.py data/boat output/boat --blend feather # or: overwrite
    python run.py data/my_photos output/mine            # your own photos (any order, any count)
    python run.py data/x output/x --planar              # planar instead of cylindrical (narrow FOV only)

Dataset: OpenCV `stitching/boat1..6.jpg` (public). Put your own photos in `data/<name>/`
(rotate the camera about its centre, ~30-50% overlap, fixed exposure/focus if possible).

## Pipeline (all in `stitcher.py`)
| Stage | Function | What we implemented |
|---|---|---|
| Features | `detect_and_describe` | OpenCV SIFT (only the detector/descriptor is a library call) |
| Matching | `match_descriptors` | Brute-force L2 via matrix form, Lowe ratio test (0.75), mutual nearest-neighbour check |
| Homography | `dlt_homography` | Hartley normalisation + DLT (SVD of 2N x 9 matrix) |
| Outliers | `ransac_homography` | Sample 4 pts, symmetric transfer error, adaptive iteration count, refit on all inliers |
| Pair validation | `pairwise_homographies` | Brown & Lowe test: inliers > 8 + 0.3 * matches |
| Ordering | `compose_to_reference` | All-pairs match graph -> max-spanning tree over inlier counts; centre image = reference; images may be unordered / unrelated ones dropped |
| Projection | `estimate_focal`, `cylindrical_warp` | Focal length from homographies (Szeliski), cylinder projection, then re-match |
| Warping | `warp_to_canvas` | Inverse mapping with bilinear sampling into a common canvas |
| Exposure | `gain_compensation` | Least-squares per-image gains over overlaps |
| Blending | `blend_feather`, `blend_multiband` | Distance-transform weights; Laplacian-pyramid multi-band blending (5 levels) |
| Cropping | `largest_inscribed_rectangle` | Largest all-valid axis-aligned rectangle (histogram stack algorithm) |

Library primitives used: `cv2.SIFT`, `cv2.remap` (sampling only), `cv2.pyrDown/pyrUp`, `cv2.distanceTransform`, numpy SVD.

## Results (boat set, 6 images, 1200px)
Pair inliers for adjacent images: 690, 445, 308, 607, 792 (of ~350-975 matches). Non-overlapping pairs are rejected (<=6 inliers).
Outputs in `output/boat/`: `matches_i_j.jpg` (green=inliers, red=outliers), `cyl_i.jpg`, `panorama_{overwrite,feather,multiband}[_cropped].jpg`.

## Discussion notes / likely questions
- **Why ratio test?** Correct match's best distance is much smaller than 2nd best; ambiguous (repetitive) features fail it.
- **Why normalise in DLT?** Pixel coordinates (~1000) vs 1 in the matrix make A ill-conditioned; normalising stabilises SVD.
- **Why RANSAC iterations adaptive?** N = log(1-p)/log(1-w^4), w = inlier ratio.
- **Why was the planar result stretched?** Homographies to a plane blow up as the view angle -> 90 deg; we first saw a 12,000 px wedge. Cylindrical projection fixes wide FOV.
- **Why multi-band?** Low frequencies (exposure) blend over wide regions, high frequencies (edges) over narrow, so no ghost/seam.
- **Limitations:** moving water/ice (the boat scene) and parallax cause slight blur at seams; no bundle adjustment so small drift accumulates in long chains; no seam finding by graph cut.
