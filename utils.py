import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


def anms(keypoints, descriptors, N=500):
    """Apply Adaptive Non-Maximal Suppression (ANMS) to keep a spatially well-distributed subset of keypoints.

    Parameters:
        keypoints: List of cv2.KeyPoint objects.
        descriptors: np.ndarray of feature descriptors aligned with keypoints.
        N: Maximum number of keypoints to retain.

    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: Filtered keypoints and their descriptors.
    """
    if len(keypoints) <= N:
        return keypoints, descriptors

    responses = np.array([kp.response for kp in keypoints])
    radii = np.full(len(keypoints), np.inf)

    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            if responses[j] > responses[i]:
                dist = (keypoints[i].pt[0] - keypoints[j].pt[0])**2 + (keypoints[i].pt[1] - keypoints[j].pt[1])**2
                if dist < radii[i]:
                    radii[i] = dist

    sorted_idx = np.argsort(-radii)
    selected_idx = sorted_idx[:N]
    keypoints_anms = [keypoints[i] for i in selected_idx]
    descriptors_anms = descriptors[selected_idx, :]
    return keypoints_anms, descriptors_anms


def plot_matches(img1, kp1, img2, kp2, matches, num_to_show=500, title="Matches"):
    """Visualize feature matches between two images.

    Parameters:
        img1: First BGR image.
        kp1: Keypoints for the first image.
        img2: Second BGR image.
        kp2: Keypoints for the second image.
        matches: Iterable of cv2.DMatch objects.
        num_to_show: Maximum number of matches to draw.
        title: Figure title.

    Returns:
        None
    """
    num_to_show = min(num_to_show, len(matches))
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:num_to_show], None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    for m in matches[:num_to_show]:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 2)

    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def match_features(des1, des2, ratio_thresh):
    """Match descriptors using BFMatcher with Lowe's ratio test.

    Parameters:
        des1: Descriptors from the query image.
        des2: Descriptors from the train image.
        ratio_thresh: Lowe ratio threshold in [0, 1]; lower is stricter.

    Returns:
        List[cv2.DMatch]: Filtered and sorted matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches_knn if m.distance < ratio_thresh * n.distance]
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    return good_matches


def _normalize_points(pts):
    """Compute Hartley normalization for 2D points.

    Parameters:
        pts: (N, 2) points.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized points and normalization matrix T.
    """
    c = pts.mean(axis=0)
    s = np.sqrt(2) / (np.mean(np.linalg.norm(pts - c, axis=1)) + 1e-8)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], float)
    pts_h = np.c_[pts, np.ones(len(pts))]
    return (T @ pts_h.T).T[:, :2], T


def compute_homography_normalized(src, dst):
    """Estimate homography using normalized DLT (Hartley normalization).

    Parameters:
        src: (N, 2) source points.
        dst: (N, 2) destination points.

    Returns:
        np.ndarray: 3x3 homography mapping src -> dst.
    """
    nsrc, Ts = _normalize_points(src)
    ndst, Td = _normalize_points(dst)
    A = []
    for (x, y), (u, v) in zip(nsrc, ndst):
        A += [[-x, -y, -1, 0, 0, 0, x * u, y * u, u],
              [0, 0, 0, -x, -y, -1, x * v, y * v, v]]
    A = np.asarray(A, float)
    _, _, VT = np.linalg.svd(A)
    Hn = VT[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def _reproj_err_sym(H, src, dst):
    """Compute symmetric reprojection error for a homography.

    Parameters:
        H: 3x3 homography mapping src -> dst.
        src: (N, 2) source points.
        dst: (N, 2) destination points.

    Returns:
        np.ndarray: Symmetric reprojection error per correspondence.
    """
    n = src.shape[0]
    src_h = np.c_[src, np.ones(n)]
    dst_h = np.c_[dst, np.ones(n)]

    p = (H @ src_h.T).T
    p = p[:, :2] / p[:, [2]]
    ef = np.linalg.norm(p - dst, axis=1)

    try:
        Hinv = np.linalg.inv(H)
        q = (Hinv @ dst_h.T).T
        q = q[:, :2] / q[:, [2]]
        eb = np.linalg.norm(q - src, axis=1)
        return 0.5 * (ef + eb)
    except np.linalg.LinAlgError:
        return ef


def ransac_homography(src_pts, dst_pts, num_iters=1000, threshold=3.0):
    """Robustly estimate a homography using RANSAC with normalized DLT and symmetric error.

    Parameters:
        src_pts: (N, 2) source points.
        dst_pts: (N, 2) destination points.
        num_iters: Number of RANSAC iterations.
        threshold: Inlier threshold in pixels for the symmetric reprojection error.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (3x3 homography, boolean inlier mask).
    """
    best_H = None
    max_inliers = 0
    best_inliers_mask = None
    n = src_pts.shape[0]

    for _ in range(num_iters):
        idx = np.random.choice(n, 4, replace=False)
        try:
            H = compute_homography_normalized(src_pts[idx], dst_pts[idx])
        except Exception:
            continue
        errors = _reproj_err_sym(H, src_pts, dst_pts)
        inliers_mask = errors < threshold
        num_inliers = int(np.sum(inliers_mask))
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers_mask = inliers_mask

    if best_inliers_mask is None or max_inliers < 4:
        raise RuntimeError("RANSAC failed to find a valid homography.")

    final_H = compute_homography_normalized(src_pts[best_inliers_mask], dst_pts[best_inliers_mask])
    return final_H, best_inliers_mask


def warp_images(img_anchor, img_left, img_right, H_left, H_right):
    """Warp left and right images to the anchor frame and compose them on a common canvas.

    Parameters:
        img_anchor: Anchor BGR image.
        img_left: Left BGR image.
        img_right: Right BGR image.
        H_left: 3x3 homography mapping left -> anchor.
        H_right: 3x3 homography mapping right -> anchor.

    Returns:
        np.ndarray: Composed panorama image.
    """
    h, w = img_anchor.shape[:2]
    corners_anchor = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    corners_left = np.array([[0, 0], [img_left.shape[1], 0], [img_left.shape[1], img_left.shape[0]], [0, img_left.shape[0]]], dtype=np.float32).reshape(-1, 1, 2)
    corners_right = np.array([[0, 0], [img_right.shape[1], 0], [img_right.shape[1], img_right.shape[0]], [0, img_right.shape[0]]], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners_left = cv2.perspectiveTransform(corners_left, H_left)
    warped_corners_right = cv2.perspectiveTransform(corners_right, H_right)
    all_corners = np.vstack((corners_anchor, warped_corners_left, warped_corners_right))

    min_x, min_y = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    max_x, max_y = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans_dist = [-min_x, -min_y]
    output_size = (max_x - min_x, max_y - min_y)

    translation_matrix = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]], dtype=float)

    result = cv2.warpPerspective(img_left, translation_matrix @ H_left, output_size)
    result[trans_dist[1]:h + trans_dist[1], trans_dist[0]:w + trans_dist[0]] = img_anchor

    result_right = cv2.warpPerspective(img_right, translation_matrix @ H_right, output_size)
    mask = (result_right > 0)
    result[mask] = result_right[mask]
    return result


def dmatches_from_H(kp1, kp2, H, max_err=3.0):
    """Synthesize cv2.DMatch objects from a homography for visualization.

    Parameters:
        kp1: List of cv2.KeyPoint in image 1.
        kp2: List of cv2.KeyPoint in image 2.
        H: 3x3 homography mapping image 1 -> image 2.
        max_err: Acceptance threshold in pixels.

    Returns:
        List[cv2.DMatch]: Matches consistent with the homography.
    """
    pts1 = np.float32([k.pt for k in kp1])
    src_h = np.hstack([pts1, np.ones((len(pts1), 1), np.float32)])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    pts2 = np.float32([k.pt for k in kp2])
    d2 = ((proj[:, None, :] - pts2[None, :, :])**2).sum(axis=2)
    j = d2.argmin(axis=1)
    min_d = np.sqrt(d2[np.arange(len(pts1)), j])
    keep = np.where(min_d <= max_err)[0]
    matches = [cv2.DMatch(_queryIdx=int(i), _trainIdx=int(j[i]), _imgIdx=0, _distance=float(min_d[i])) for i in keep]
    return matches


def plot_correspondences(img1, pts1, img2, pts2, title="Correspondencias manuales",
                         radius=10, line_th=3, show_index=True, scale=1.0):
    """Plot corresponding point pairs between two images.

    Parameters:
        img1: First BGR or grayscale image.
        pts1: (N, 2) points in image 1.
        img2: Second BGR or grayscale image.
        pts2: (N, 2) matching points in image 2.
        title: Figure title.
        radius: Marker radius for points.
        line_th: Line thickness for connections.
        show_index: Whether to draw point indices.
        scale: Optional scale factor for display.

    Returns:
        None
    """
    if img1.ndim == 2:
        img1c = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1c = img1.copy()
    if img2.ndim == 2:
        img2c = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2c = img2.copy()

    h1, w1 = img1c.shape[:2]
    h2, w2 = img2c.shape[:2]
    Hc = max(h1, h2)
    Wc = w1 + w2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1c
    canvas[:h2, w1:w1 + w2] = img2c
    out = canvas.copy()

    n = min(len(pts1), len(pts2))
    if n == 0:
        print("⚠️ No hay puntos para dibujar.")
        return

    for i in range(n):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)) + w1, int(round(y2)))

        cv2.circle(out, p1, radius + 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p1, radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p2, radius + 3, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, p2, radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(out, p1, p2, (255, 0, 0), thickness=line_th, lineType=cv2.LINE_AA)

        if show_index:
            pos1 = (p1[0] + radius + 6, p1[1] - radius - 6)
            pos2 = (p2[0] + radius + 6, p2[1] - radius - 6)
            for img_pos in (pos1, pos2):
                cv2.putText(out, str(i + 1), img_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, str(i + 1), img_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def pick_points_cv(img_bgr, n=6, win='pick', scale=1.6, marker_size=32, thickness=5, color=(0, 255, 255),
                   tilted=False, show_index=True):
    """Interactively pick points on an image using OpenCV mouse callbacks.

    Parameters:
        img_bgr: BGR image to click on.
        n: Number of points to collect.
        win: Window name.
        scale: Display scale factor.
        marker_size: Marker size for drawn points.
        thickness: Marker line thickness.
        color: Marker color (BGR).
        tilted: Use tilted cross marker if True.
        show_index: Whether to overlay point indices.

    Returns:
        np.ndarray: (n, 2) array of clicked coordinates (x, y).
    """
    disp = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    pts_disp, pts = [], []
    marker_type = cv2.MARKER_TILTED_CROSS if tilted else cv2.MARKER_CROSS

    def cb(event, x, y, flags, param):
        nonlocal pts, pts_disp
        if event == cv2.EVENT_LBUTTONDOWN:
            pts_disp.append((x, y))
            pts.append((x / scale, y / scale))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, cb)

    while True:
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            pts = []
            pts_disp = []
            break

        vis = disp.copy()
        for i, (x, y) in enumerate(pts_disp):
            x, y = int(x), int(y)
            cv2.drawMarker(vis, (x, y), (0, 0, 0), markerType=marker_type, markerSize=marker_size + 6,
                           thickness=thickness + 3, line_type=cv2.LINE_AA)
            cv2.drawMarker(vis, (x, y), color, markerType=marker_type, markerSize=marker_size,
                           thickness=thickness, line_type=cv2.LINE_AA)
            if show_index:
                pos = (x + marker_size // 2 + 8, y - marker_size // 2 - 8)
                cv2.putText(vis, str(i + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, str(i + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and pts:
            pts.pop()
            pts_disp.pop()
        if (k in (13, 10, 32)) and len(pts) >= n:
            break
        if k == 27:
            pts = []
            pts_disp = []
            break

    cv2.setMouseCallback(win, lambda *args: None)
    cv2.destroyWindow(win)
    for _ in range(3):
        cv2.waitKey(1)
        time.sleep(0.01)
    return np.array(pts, dtype=np.float32)


def warp_to_anchor_canvas(img_anchor, img_left, img_right, H_L2A, H_R2A):
    """Warp three images to a common canvas in the anchor coordinate system.

    Parameters:
        img_anchor: Anchor BGR image.
        img_left: Left BGR image.
        img_right: Right BGR image.
        H_L2A: 3x3 homography mapping left -> anchor.
        H_R2A: 3x3 homography mapping right -> anchor.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (left_warped, anchor_warped, right_warped).
    """
    hA, wA = img_anchor.shape[:2]
    hL, wL = img_left.shape[:2]
    hR, wR = img_right.shape[:2]

    cornA = np.float32([[0, 0], [wA, 0], [wA, hA], [0, hA]]).reshape(-1, 1, 2)
    cornL = np.float32([[0, 0], [wL, 0], [wL, hL], [0, hL]]).reshape(-1, 1, 2)
    cornR = np.float32([[0, 0], [wR, 0], [wR, hR], [0, hR]]).reshape(-1, 1, 2)

    cornL_A = cv2.perspectiveTransform(cornL, H_L2A)
    cornR_A = cv2.perspectiveTransform(cornR, H_R2A)
    all_pts = np.vstack([cornA, cornL_A, cornR_A]).reshape(-1, 2)

    min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)

    tx, ty = -min(0, min_x), -min(0, min_y)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    W = int(max_x + tx)
    Hc = int(max_y + ty)

    left_warped = cv2.warpPerspective(img_left, T @ H_L2A, (W, Hc))
    right_warped = cv2.warpPerspective(img_right, T @ H_R2A, (W, Hc))
    anchor_warped = cv2.warpPerspective(img_anchor, T, (W, Hc))
    return left_warped, anchor_warped, right_warped


def feather_blend(images, gamma=1.0):
    """Blend aligned images using distance-transform weights (feathering).

    Parameters:
        images: List of BGR images already aligned on the same canvas.
        gamma: Exponent applied to distance weights; higher emphasizes interiors.

    Returns:
        np.ndarray: Blended BGR image.
    """
    Hc, Wc = images[0].shape[:2]
    weights = []
    for img in images:
        mask = (img.sum(axis=2) > 0).astype(np.uint8)
        mask_u8 = (mask * 255).astype(np.uint8)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
        dist *= mask
        if gamma != 1.0:
            dist = dist ** gamma
        weights.append(dist)

    weights[1] *= 1.3
    eps = 1e-8
    Wsum = np.sum(weights, axis=0) + eps
    out = np.zeros_like(images[0], dtype=np.float32)
    for img, Wm in zip(images, weights):
        out += img.astype(np.float32) * Wm[..., None]
    out = (out / Wsum[..., None]).clip(0, 255).astype(np.uint8)
    return out


def match_exposure(imgA, imgB):
    """Match exposure of imgB to imgA over their overlapping region using per-channel medians.

    Parameters:
        imgA: Reference BGR image.
        imgB: Target BGR image to be adjusted.

    Returns:
        np.ndarray: Exposure-adjusted image.
    """
    mA = (imgA.sum(axis=2) > 0)
    mB = (imgB.sum(axis=2) > 0)
    overlap = mA & mB
    if overlap.sum() < 100:
        return imgB
    A_med = np.median(imgA[overlap], axis=0)
    B_med = np.median(imgB[overlap], axis=0)
    gain = (A_med + 1e-6) / (B_med + 1e-6)
    return (imgB.astype(np.float32) * gain).clip(0, 255).astype(np.uint8)
