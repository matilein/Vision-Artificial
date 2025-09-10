import numpy as np
import cv2
import matplotlib.pyplot as plt
def anms(keypoints, descriptors, N=500):
    """
    Adaptive Non-Maximal Suppression (ANMS)
    keypoints: lista de cv2.KeyPoint
    descriptors: array de descriptores asociado a esos keypoints
    N: número máximo de puntos a retener
    """
    if len(keypoints) <= N:
        return keypoints, descriptors

    # Respuesta de cada keypoint (cuánto "fuerte" es)
    responses = np.array([kp.response for kp in keypoints])

    # Inicializamos radio supresión con infinito
    radii = np.full(len(keypoints), np.inf)

    # Para cada punto calculamos la distancia mínima a otro con mayor respuesta
    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            if responses[j] > responses[i]:
                dist = (keypoints[i].pt[0] - keypoints[j].pt[0])**2 + \
                       (keypoints[i].pt[1] - keypoints[j].pt[1])**2
                if dist < radii[i]:
                    radii[i] = dist

    # Ordenamos por radio descendente
    sorted_idx = np.argsort(-radii)
    selected_idx = sorted_idx[:N]

    # Filtramos keypoints y descriptores
    keypoints_anms = [keypoints[i] for i in selected_idx]
    descriptors_anms = descriptors[selected_idx, :]

    return keypoints_anms, descriptors_anms



def plot_matches(img1, kp1, img2, kp2, matches, num_to_show=500, title="Matches"):
    num_to_show = min(num_to_show, len(matches))

    # usar drawMatches para crear imagen base
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:num_to_show], None,
        matchColor=(0, 255, 0),  # verde
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # redibujar líneas más gruesas encima en rojo
    for m in matches[:num_to_show]:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 2)  # rojo, grosor 2 px

    # mostrar con matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def match_features(des1, des2, ratio_thresh):
    """
    Matching con BFMatcher + Lowe's ratio test.
    des1: descriptores de la imagen origen
    des2: descriptores de la imagen destino
    ratio_thresh: umbral del test de Lowe (0.65–0.7 para DLT, 0.75–0.8 para RANSAC)
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_knn = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # ordenar matches por distancia (los mejores primero)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    return good_matches


def compute_homography(ori, dst):
    """
    Calcula homografía H tal que dst ≈ H * ori
    usando el algoritmo DLT (Direct Linear Transform).
    ori: puntos fuente (Nx2)
    dst: puntos destino (Nx2)
    """
    assert ori.shape[0] == dst.shape[0] and ori.shape[0] >= 4, \
        "Se necesitan al menos 4 correspondencias"

    A = []
    for i in range(len(ori)):
        x, y = ori[i]
        xp, yp = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.array(A)

    # Resolver A h = 0 usando SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]             # vector asociado al menor valor singular
    h = h / h[-1]             # normalizar para que h[8] = 1

    H = h.reshape(3, 3)
    return H

def ransac_homography(src_pts, dst_pts, num_iters=1000, threshold=3.0):
    best_H = None
    max_inliers = 0
    best_inliers_mask = None

    n = src_pts.shape[0]

    for _ in range(num_iters):
        # seleccionar 4 pares random
        idx = np.random.choice(n, 4, replace=False)
        H = compute_homography(src_pts[idx], dst_pts[idx])

        # proyectar src_pts con H
        src_homo = np.hstack([src_pts, np.ones((n,1))])
        proj = (H @ src_homo.T).T
        proj = proj[:, :2] / proj[:, 2, np.newaxis]

        # calcular error
        errors = np.linalg.norm(dst_pts - proj, axis=1)
        inliers_mask = errors < threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers_mask = inliers_mask

    # recalcular H usando todos los inliers
    final_H = compute_homography(src_pts[best_inliers_mask], dst_pts[best_inliers_mask])
    return final_H, best_inliers_mask

def warp_images(img_anchor, img_left, img_right, H_left, H_right):
    """
    Warpea y compone las imágenes izquierda y derecha hacia el marco de la ancla.
    Devuelve la panorámica final.
    """

    h, w = img_anchor.shape[:2]

    # --- Esquinas de cada imagen ---
    corners_anchor = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    corners_left   = np.array([[0,0],[img_left.shape[1],0],[img_left.shape[1],img_left.shape[0]],[0,img_left.shape[0]]], dtype=np.float32).reshape(-1,1,2)
    corners_right  = np.array([[0,0],[img_right.shape[1],0],[img_right.shape[1],img_right.shape[0]],[0,img_right.shape[0]]], dtype=np.float32).reshape(-1,1,2)

    # Transformar esquinas con las homografías
    warped_corners_left  = cv2.perspectiveTransform(corners_left,  H_left)
    warped_corners_right = cv2.perspectiveTransform(corners_right, H_right)

    all_corners = np.vstack((corners_anchor, warped_corners_left, warped_corners_right))

    [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # --- Tamaño del canvas ---
    trans_dist = [-min_x, -min_y]
    output_size = (max_x - min_x, max_y - min_y)

    # --- Matriz de traslación ---
    translation_matrix = np.array([
        [1, 0, trans_dist[0]],
        [0, 1, trans_dist[1]],
        [0, 0, 1]
    ])

    # --- Warp de cada imagen ---
    result = cv2.warpPerspective(img_left, translation_matrix @ H_left, output_size)
    result[trans_dist[1]:h+trans_dist[1], trans_dist[0]:w+trans_dist[0]] = img_anchor

    result_right = cv2.warpPerspective(img_right, translation_matrix @ H_right, output_size)
    mask = (result_right > 0)
    result[mask] = result_right[mask]

    return result