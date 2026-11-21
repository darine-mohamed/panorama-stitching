"""Usage: python run.py data/boat output/boat  [--blend multiband|feather|overwrite]"""
import argparse, os, time
import cv2
import stitcher as S

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("out")
ap.add_argument("--blend", default="multiband")
ap.add_argument("--max-dim", type=int, default=1200)
ap.add_argument("--ratio", type=float, default=0.75)
ap.add_argument("--thresh", type=float, default=3.0)
ap.add_argument("--planar", action="store_true")
ap.add_argument("--no-gain", action="store_true")
a = ap.parse_args()

files, imgs = S.load_images(a.folder, a.max_dim)
print("images:", [os.path.basename(f) for f in files])
t = time.time()
r = S.stitch(imgs, a.out, a.ratio, a.thresh, a.blend, not a.no_gain, cylindrical=not a.planar)
print(f"done in {time.time()-t:.1f}s")
cv2.imwrite(os.path.join(a.out, f"panorama_{a.blend}.jpg"), r["pano"])
if "cropped" in r:
    cv2.imwrite(os.path.join(a.out, f"panorama_{a.blend}_cropped.jpg"), r["cropped"])
