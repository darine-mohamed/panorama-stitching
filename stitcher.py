"""
Panorama stitching pipeline (written from scratch on top of numpy + OpenCV primitives).

Stages (each is its own function so every team member can explain it):
  1. detect_and_describe   - SIFT keypoints + 128-d descriptors   (OpenCV SIFT)
  2. match_descriptors     - brute-force kNN + Lowe ratio test + mutual check (OURS)
  3. estimate_homography   - normalised DLT + RANSAC + inlier refit            (OURS)
  4. build_graph / order   - image-ordering via match graph + spanning tree    (OURS)
  5. warp_to_canvas        - inverse warping with bilinear sampling            (OURS, cv2.remap only samples)
  6. blend                 - gain compensation, feather, multi-band blending   (OURS)
  7. crop_black_borders    - largest axis-aligned valid rectangle              (OURS)
"""
import glob
import os
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 1. Features
# ----------------------------------------------------------------------------
def detect_and_describe(img_bgr, n_features=4000):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kps, desc = sift.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kps])
    return pts, desc


# ----------------------------------------------------------------------------
# 2. Matching
# ----------------------------------------------------------------------------
def match_descriptors(d1, d2, ratio=0.75, mutual=True):
    """Return (M,2) int array of index pairs (i in d1, j in d2).

    Euclidean distances via |a-b|^2 = |a|^2 + |b|^2 - 2ab.
    Lowe ratio test: keep match if best_dist < ratio * second_best_dist.
    Mutual check: j's nearest neighbour in d1 must also be i.
    """
    d1 = d1.astype(np.float32)
    d2 = d2.astype(np.float32)
    D = (d1 ** 2).sum(1)[:, None] + (d2 ** 2).sum(1)[None, :] - 2 * d1 @ d2.T
    D = np.sqrt(np.maximum(D, 0))

    # two smallest per row
    idx = np.argpartition(D, 1, axis=1)[:, :2]
    rows = np.arange(D.shape[0])
    a, b = D[rows, idx[:, 0]], D[rows, idx[:, 1]]
    best = np.where(a <= b, idx[:, 0], idx[:, 1])
    d_best = np.minimum(a, b)
    d_second = np.maximum(a, b)
    keep = d_best < ratio * d_second

    if mutual:
        back = D.argmin(0)  # nearest in d1 for every d2
        keep &= back[best] == rows

    i = np.where(keep)[0]
    return np.stack([i, best[i]], 1)


# ----------------------------------------------------------------------------
# 3. Homography: DLT + RANSAC
# ----------------------------------------------------------------------------
def _normalise(pts):
    """Hartley normalisation: centroid at origin, mean distance sqrt(2)."""
    c = pts.mean(0)
    s = np.sqrt(2) / (np.linalg.norm(pts - c, axis=1).mean() + 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, T


def dlt_homography(src, dst):
    """Direct Linear Transform: solve A h = 0 with SVD, N >= 4 points. dst ~ H src."""
    s, Ts = _normalise(src)
    d, Td = _normalise(dst)
    n = len(s)
    A = np.zeros((2 * n, 9))
    x, y, u, v = s[:, 0], s[:, 1], d[:, 0], d[:, 1]
    A[0::2, 0:3] = np.stack([x, y, np.ones(n)], 1)
    A[0::2, 6:9] = -u[:, None] * np.stack([x, y, np.ones(n)], 1)
    A[1::2, 3:6] = np.stack([x, y, np.ones(n)], 1)
    A[1::2, 6:9] = -v[:, None] * np.stack([x, y, np.ones(n)], 1)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def ransac_homography(src, dst, thresh=3.0, iters=5000, conf=0.995, seed=0):
    """Adaptive RANSAC. Error = symmetric transfer distance (forward + backward)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 4:
        return None, np.zeros(n, bool)
    best_inl, best_cnt = np.zeros(n, bool), 0
    N = iters
    it = 0
    while it < N:
        it += 1
        samp = rng.choice(n, 4, replace=False)
        try:
            H = dlt_homography(src[samp], dst[samp])
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(H).all():
            continue
        e = np.linalg.norm(apply_h(H, src) - dst, axis=1) + np.linalg.norm(apply_h(Hi, dst) - src, axis=1)
        inl = e < 2 * thresh
        cnt = inl.sum()
        if cnt > best_cnt:
            best_cnt, best_inl = cnt, inl
            w = cnt / n  # adaptive iteration count: N = log(1-p)/log(1-w^4)
            if w > 0:
                N = min(iters, int(np.log(1 - conf) / np.log(1 - w ** 4 + 1e-12)) + 1)
    if best_cnt < 4:
        return None, best_inl
    # refit on all inliers, then re-evaluate once
    H = dlt_homography(src[best_inl], dst[best_inl])
    Hi = np.linalg.inv(H)
    e = np.linalg.norm(apply_h(H, src) - dst, axis=1) + np.linalg.norm(apply_h(Hi, dst) - src, axis=1)
    inl = e < 2 * thresh
    if inl.sum() >= 4:
        H = dlt_homography(src[inl], dst[inl])
    return H, inl


# ----------------------------------------------------------------------------
# 4. Pairwise matching graph -> ordering / reference frame
# ----------------------------------------------------------------------------
def pairwise_homographies(feats, ratio=0.75, thresh=3.0, min_inliers=25, log=print):
    """H[(i,j)] maps points of image j into image i's frame (i<j). Also returns inlier data."""
    n = len(feats)
    result = {}
    for i in range(n):
        for j in range(i + 1, n):
            m = match_descriptors(feats[i][1], feats[j][1], ratio)
            if len(m) < 8:
                continue
            p_i, p_j = feats[i][0][m[:, 0]], feats[j][0][m[:, 1]]
            H, inl = ransac_homography(p_j, p_i, thresh)
            if H is None:
                continue
            # Brown & Lowe confidence check: inliers > 8 + 0.3 * matches
            ok = inl.sum() >= max(min_inliers, 8 + 0.3 * len(m))
            log(f"  pair ({i},{j}): {len(m):4d} matches, {inl.sum():4d} inliers -> {'OK' if ok else 'reject'}")
            if ok:
                result[(i, j)] = dict(H=H, matches=m, inliers=inl, n_inl=int(inl.sum()))
    return result


def compose_to_reference(n, pairs):
    """Maximum-spanning-tree over inlier counts; reference = most connected image
    (closest to the 'middle' to limit perspective stretch). Returns (ref, H_to_ref list)."""
    w = {}
    for (i, j), d in pairs.items():
        w.setdefault(i, {})[j] = (d["n_inl"], np.linalg.inv(d["H"]))  # i -> j frame
        w.setdefault(j, {})[i] = (d["n_inl"], d["H"])                  # j -> i frame
    if not w:
        raise RuntimeError("No image pairs could be matched.")
    # reference: image minimising max hop-distance to the others in the graph
    def ecc(r):
        dist, q = {r: 0}, [r]
        for u in q:
            for v in w.get(u, {}):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return (max(dist.values()), -len(dist)) if len(dist) == len(w) else (99, -len(dist))
    ref = min(w, key=ecc)
    # Prim's algorithm on max-weight edges, growing from ref
    Hs = {ref: np.eye(3)}
    while True:
        best = None
        for u in Hs:
            for v, (wt, Hvu) in w[u].items():  # Hvu maps frame-v points into frame-u
                if v not in Hs and (best is None or wt > best[0]):
                    best = (wt, u, v, Hvu)
        if best is None:
            break
        _, u, v, Hvu = best
        Hs[v] = Hs[u] @ Hvu
    return ref, [Hs.get(i) for i in range(n)]



# ----------------------------------------------------------------------------
# 4b. Focal length from homographies + cylindrical projection
# ----------------------------------------------------------------------------
def focal_from_homography(H, shape_i, shape_j):
    """Szeliski: for H = K_i R K_j^-1 with zero skew / centred principal point, f is
    recoverable from the rotation's orthonormality. Returns (f_i, f_j) estimates or None."""
    def shift(shape):
        h, w = shape
        return np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.]])
    # H maps image j -> image i.  Centre both coordinate systems.
    Hc = shift(shape_i) @ H @ np.linalg.inv(shift(shape_j))
    Hc = Hc / Hc[2, 2]
    h = Hc.ravel()
    out = []
    # focal of the source (j)
    d1, d2 = h[6] * h[7], (h[7] - h[6]) * (h[7] + h[6])
    v1, v2 = -(h[0] * h[1] + h[3] * h[4]) / d1 if abs(d1) > 1e-12 else -1, \
             (h[0] ** 2 + h[3] ** 2 - h[1] ** 2 - h[4] ** 2) / d2 if abs(d2) > 1e-12 else -1
    out.append(v1 if v1 > 0 else (v2 if v2 > 0 else None))
    # focal of the destination (i)
    d1, d2 = h[0] * h[3] + h[1] * h[4], h[0] ** 2 + h[1] ** 2 - h[3] ** 2 - h[4] ** 2
    v1, v2 = -h[2] * h[5] / d1 if abs(d1) > 1e-12 else -1, \
             (h[5] ** 2 - h[2] ** 2) / d2 if abs(d2) > 1e-12 else -1
    out.append(v1 if v1 > 0 else (v2 if v2 > 0 else None))
    return [np.sqrt(v) for v in out if v is not None]


def estimate_focal(pairs, shapes):
    fs = []
    for (i, j), d in pairs.items():
        fs += focal_from_homography(d["H"], shapes[i], shapes[j])
    fs = [f for f in fs if 200 < f < 20000]
    return float(np.median(fs)) if fs else float(max(shapes[0]))


def cylindrical_warp(img, f):
    """Project onto a cylinder of radius f: output (u,v) = (f*theta, f*h)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    half = f * np.arctan(cx / f)            # half width of output
    W = int(np.ceil(2 * half))
    us, vs = np.meshgrid(np.arange(W) - half, np.arange(h) - cy)
    theta, hh = us / f, vs / f
    x = f * np.tan(theta) + cx
    y = f * hh / np.cos(theta) + cy
    out = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    return out

# ----------------------------------------------------------------------------
# 5. Warping
# ----------------------------------------------------------------------------
def canvas_geometry(shapes, Hs, max_pixels=5e6):
    corners = []
    for (h, w), H in zip(shapes, Hs):
        if H is None:
            continue
        c = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        corners.append(apply_h(H, c))
    corners = np.vstack(corners)
    xmin, ymin = np.floor(corners.min(0)).astype(int)
    xmax, ymax = np.ceil(corners.max(0)).astype(int)
    W, Hh = xmax - xmin, ymax - ymin
    scale = 1.0
    if W * Hh > max_pixels:
        scale = np.sqrt(max_pixels / (W * Hh))
    T = np.array([[scale, 0, -xmin * scale], [0, scale, -ymin * scale], [0, 0, 1]])
    return T, (int(np.ceil(W * scale)), int(np.ceil(Hh * scale)))


def warp_to_canvas(img, H_full, size):
    """Inverse warping: for every canvas pixel q, sample img at H^-1 q (bilinear).
    Returns warped image (float32) and validity mask."""
    Wc, Hc = size
    Hinv = np.linalg.inv(H_full)
    ys, xs = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    q = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size, np.float32)])
    p = Hinv @ q
    mapx = (p[0] / p[2]).reshape(Hc, Wc).astype(np.float32)
    mapy = (p[1] / p[2]).reshape(Hc, Wc).astype(np.float32)
    warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    h, w = img.shape[:2]
    mask = ((mapx >= 0) & (mapx <= w - 1) & (mapy >= 0) & (mapy <= h - 1) & (p[2].reshape(Hc, Wc) > 0))
    return warped, mask


# ----------------------------------------------------------------------------
# 6. Blending
# ----------------------------------------------------------------------------
def gain_compensation(warped, masks, sigma_n=10.0, sigma_g=0.1):
    """Brown & Lowe: one gain per image minimising intensity differences in overlaps."""
    n = len(warped)
    A, b = np.zeros((n, n)), np.zeros(n)
    I = [w.mean(2) for w in warped]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ov = masks[i] & masks[j]
            N = ov.sum()
            if N == 0:
                continue
            Ii, Ij = I[i][ov].mean(), I[j][ov].mean()
            A[i, i] += N * (2 * Ii ** 2 / sigma_n ** 2 + 1 / sigma_g ** 2)
            A[i, j] -= 2 * N * Ii * Ij / sigma_n ** 2
            b[i] += N / sigma_g ** 2
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.ones(n)


def weight_map(mask):
    """Distance to nearest invalid pixel (grows to image centre) -> feather weight."""
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return d / (d.max() + 1e-9)


def blend_overwrite(warped, masks):
    out = np.zeros_like(warped[0])
    for w, m in zip(warped, masks):
        out[m] = w[m]
    return out


def blend_feather(warped, masks):
    ws = [weight_map(m) for m in masks]
    num = sum(w[..., None] * im for w, im in zip(ws, warped))
    den = sum(ws)[..., None] + 1e-9
    return num / den


def blend_multiband(warped, masks, levels=5):
    """Multi-band (Laplacian pyramid) blending. Each pixel takes the label of the image
    whose feather weight is largest ('max-weight' seam); masks are blurred at coarser levels
    so low frequencies blend over wide areas and high frequencies over narrow ones."""
    ws = np.stack([weight_map(m) for m in masks])           # (n,H,W)
    lab = ws.argmax(0)
    hard = np.stack([(lab == k) & masks[k] for k in range(len(masks))]).astype(np.float32)

    def gauss_pyr(x):
        p = [x]
        for _ in range(levels):
            x = cv2.pyrDown(x)
            p.append(x)
        return p

    def lap_pyr(x):
        g = gauss_pyr(x)
        return [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(levels)] + [g[-1]]

    blended = None
    wsum = [gauss_pyr(h) for h in hard]
    for k, im in enumerate(warped):
        im = im.copy()
        im[~masks[k]] = 0
        L = lap_pyr(im)
        contrib = [L[l] * wsum[k][l][..., None] for l in range(levels + 1)]
        blended = contrib if blended is None else [a + b for a, b in zip(blended, contrib)]
    norm = [sum(wsum[k][l] for k in range(len(warped))) + 1e-9 for l in range(levels + 1)]
    layers = [blended[l] / norm[l][..., None] for l in range(levels + 1)]
    out = layers[-1]
    for l in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(layers[l].shape[1], layers[l].shape[0])) + layers[l]
    return np.clip(out, 0, 255)


# ----------------------------------------------------------------------------
# 7. Cropping
# ----------------------------------------------------------------------------
def largest_inscribed_rectangle(mask):
    """Largest axis-aligned all-True rectangle (histogram / stack algorithm, O(HW))."""
    H, W = mask.shape
    heights = np.zeros(W, int)
    best = (0, 0, 0, 0, 0)  # area, x0, y0, x1, y1
    for y in range(H):
        heights = np.where(mask[y], heights + 1, 0)
        stack = []
        for x in range(W + 1):
            h = heights[x] if x < W else 0
            start = x
            while stack and stack[-1][1] >= h:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, sx, y - sh + 1, x, y + 1)
                start = sx
            stack.append((start, h))
    _, x0, y0, x1, y1 = best
    return x0, y0, x1, y1


# ----------------------------------------------------------------------------
# Visualisation helpers
# ----------------------------------------------------------------------------
def draw_matches(img1, img2, p1, p2, inliers=None, max_draw=150, seed=0):
    h = max(img1.shape[0], img2.shape[0])
    canvas = np.zeros((h, img1.shape[1] + img2.shape[1], 3), np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2
    idx = np.arange(len(p1))
    rng = np.random.default_rng(seed)
    if len(idx) > max_draw:
        idx = rng.choice(idx, max_draw, replace=False)
    for k in idx:
        good = True if inliers is None else bool(inliers[k])
        col = (0, 255, 0) if good else (0, 0, 255)
        a = tuple(int(v) for v in p1[k])
        b = (int(p2[k][0] + img1.shape[1]), int(p2[k][1]))
        cv2.line(canvas, a, b, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, col, -1)
        cv2.circle(canvas, b, 3, col, -1)
    return canvas


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------
def load_images(folder, max_dim=1200):
    # set() + normcase: Windows globbing is case-insensitive, so "*.jpg" and "*.JPG"
    # would otherwise return every file twice.
    found = {os.path.normcase(os.path.abspath(f))
             for e in ("*.jpg", "*.jpeg", "*.png", "*.JPG")
             for f in glob.glob(os.path.join(folder, e))}
    files = sorted(found)
    imgs = []
    for f in files:
        im = cv2.imread(f)
        s = max_dim / max(im.shape[:2])
        if s < 1:
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        imgs.append(im)
    return files, imgs


def stitch(imgs, out_dir=None, ratio=0.75, ransac_thresh=3.0, blend="multiband", gain=True, crop=True, cylindrical=True, log=print):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    log(f"[1] detecting features in {len(imgs)} images")
    feats = [detect_and_describe(im) for im in imgs]
    for i, (p, _) in enumerate(feats):
        log(f"  image {i}: {len(p)} keypoints")

    log("[2-3] matching + RANSAC for all pairs")
    pairs = pairwise_homographies(feats, ratio, ransac_thresh, log=log)
    if out_dir:
        for (i, j), d in pairs.items():
            m = d["matches"]
            vis = draw_matches(imgs[i], imgs[j], feats[i][0][m[:, 0]], feats[j][0][m[:, 1]], d["inliers"])
            cv2.imwrite(os.path.join(out_dir, f"matches_{i}_{j}.jpg"), vis)

    if cylindrical:
        f = estimate_focal(pairs, [im.shape[:2] for im in imgs])
        log(f"[3b] estimated focal length = {f:.1f}px -> cylindrical projection, re-matching")
        imgs = [cylindrical_warp(im, f) for im in imgs]
        feats = [detect_and_describe(im) for im in imgs]
        pairs = pairwise_homographies(feats, ratio, ransac_thresh, log=log)
        if out_dir:
            for i, im in enumerate(imgs):
                cv2.imwrite(os.path.join(out_dir, f"cyl_{i}.jpg"), im)
    ref, Hs = compose_to_reference(len(imgs), pairs)
    used = [i for i, H in enumerate(Hs) if H is not None]
    log(f"[4] reference image = {ref}; stitched images = {used}; dropped = {sorted(set(range(len(imgs))) - set(used))}")

    T, size = canvas_geometry([im.shape[:2] for im in imgs], Hs)
    log(f"[5] canvas size = {size}")
    warped, masks = [], []
    for i in used:
        w, m = warp_to_canvas(imgs[i], T @ Hs[i], size)
        warped.append(w)
        masks.append(m)

    if gain:
        g = gain_compensation(warped, masks)
        log(f"[6] gains: {np.round(g, 3)}")
        warped = [w * gi for w, gi in zip(warped, g)]

    blenders = dict(overwrite=blend_overwrite, feather=blend_feather, multiband=blend_multiband)
    pano = np.clip(blenders[blend](warped, masks), 0, 255).astype(np.uint8)
    union = np.any(masks, 0)
    results = dict(pano=pano, union=union, warped=warped, masks=masks)

    if crop:
        # erode a little so interpolated edge pixels don't leak into the crop
        safe = cv2.erode(union.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        x0, y0, x1, y1 = largest_inscribed_rectangle(safe)
        results["cropped"] = pano[y0:y1, x0:x1]
        log(f"[7] cropped to ({x0},{y0})-({x1},{y1})")
    return results
