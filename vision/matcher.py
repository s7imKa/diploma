from dataclasses import dataclass
from typing import Optional, Sequence
import cv2
import numpy as np

from .feature_strategies import FeatureStrategy, ORBStrategy, SIFTStrategy


@dataclass
class MatchResult:
    matched_image: Optional[np.ndarray]
    heatmap: Optional[np.ndarray]
    homography: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    similarity: float  # main metric: inliers / min(kp)
    similarity_good: float  # good matches / min(kp)
    inliers: int
    reprojection_error: float
    adaptive_threshold: float
    stability: float
    keypoints1: int
    keypoints2: int
    good_matches: int
    method: str
    # New fields for UX improvements
    verdict: str  # "Object Found" | "Partial Match" | "Object Not Found"
    explanation: str  # User-friendly explanation of result
    is_homography_valid: bool  # True if inliers >= 8 for stable homography
    match_time_ms: float = 0.0
    start_tick: float = 0.0
    end_tick: float = 0.0
    tick_diff: float = 0.0
    ransac_vis: Optional[np.ndarray] = None  # До/після RANSAC для рисунка 3.4


class ImageMatcher:
    def __init__(
        self,
        strategy: Optional[FeatureStrategy] = None,
        ratio_thresh: float = 0.75,
        use_knn: bool = True,
        cross_check: bool = False,
        apply_blur: bool = False,
        blur_kernel: int = 5,
    ):
        # Strategy allows switching ORB/AKAZE/SIFT
        self.strategy: FeatureStrategy = strategy or ORBStrategy()
        self.ratio_thresh = ratio_thresh
        self.use_knn = use_knn
        self.cross_check = cross_check
        self.apply_blur = apply_blur
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.detector = self.strategy.create_detector()
        self.matcher = self.strategy.create_matcher(cross_check=cross_check)

    def set_strategy(self, strategy: FeatureStrategy):
        self.strategy = strategy
        self.detector = self.strategy.create_detector()
        self.matcher = self.strategy.create_matcher(cross_check=self.cross_check)

    def set_matcher_mode(self, use_knn: bool, ratio_thresh: float, cross_check: bool):
        self.use_knn = use_knn
        self.ratio_thresh = ratio_thresh
        self.cross_check = cross_check
        self.matcher = self.strategy.create_matcher(cross_check=cross_check)

    def set_preprocessing(self, apply_blur: bool, blur_kernel: int):
        self.apply_blur = apply_blur
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.apply_blur:
            gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        return gray

    def detect_and_match(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        import time
        t0 = time.perf_counter()
        gray1 = self._preprocess(img1)
        gray2 = self._preprocess(img2)

        keypoints1, descriptors1 = self.detector.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.detector.detectAndCompute(gray2, None)

        if descriptors1 is None or descriptors2 is None or len(keypoints1) == 0 or len(keypoints2) == 0:
            t1 = time.perf_counter()
            return MatchResult(
                None, None, None, None, 0.0, 0.0, 0, 0.0, 0.0, 0.0, len(keypoints1), len(keypoints2), 0, self.strategy.name,
                "🔴 Обʼєкт не знайдено", "Недостатньо ключових точок на одному з зображень", False,
                match_time_ms=(t1-t0)*1000
            )

        if self.use_knn and not self.cross_check:
            raw_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            good = self._ratio_filter(raw_matches)
        else:
            # crossCheck mode gives one best match each, no ratio test
            raw_matches = self.matcher.match(descriptors1, descriptors2)
            good = sorted(raw_matches, key=lambda m: m.distance)

        if len(good) < 4:
            t1 = time.perf_counter()
            return MatchResult(
                None, None, None, None, 0.0, 0.0, 0, 0.0, 0.0, 0.0, len(keypoints1), len(keypoints2), len(good), self.strategy.name,
                "🔴 Обʼєкт не знайдено", f"Недостатньо збігів: {len(good)} (потрібно ≥ 4)", False,
                match_time_ms=(t1-t0)*1000
            )

        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        similarity = 0.0
        similarity_good = 0.0
        reprojection_error = 0.0
        stability = 0.0
        verdict = "🔴 Обʼєкт не знайдено"
        explanation = "Не вдалося обчислити гомографію"
        is_homography_valid = False

        if homography is not None and mask is not None and mask.size > 0:
            min_kp = max(1, min(len(keypoints1), len(keypoints2)))
            similarity = 100.0 * inliers / min_kp
            similarity_good = 100.0 * len(good) / min_kp
            reprojection_error = self._reprojection_error(src_pts, dst_pts, homography, mask)
            stability = 1.0 / (1.0 + reprojection_error)

            # ADAPTIVE FALLBACK: if inliers < 8, homography is unstable
            if inliers >= 8:
                is_homography_valid = True
                if similarity >= 30.0:  # Strong match
                    verdict = "🟢 Обʼєкт знайдено"
                    explanation = f"Стабільна гомографія з {inliers} інлайєрами (схожість {similarity:.1f}%)"
                else:
                    verdict = "🟡 Часткова схожість"
                    explanation = f"Недостатня схожість: {similarity:.1f}% (потрібно ≥ 30%)"
            else:
                # Not enough inliers for stable homography - show only matches
                verdict = "🟡 Часткова схожість"
                explanation = f"Недостатньо інлайєрів для стабільної гомографії ({inliers} < 8)"
                homography = None  # Disable homography drawing
                mask = None

        adaptive_threshold = self._adaptive_threshold(similarity, similarity_good, len(good))

        matched_vis = self._draw_matches(img1, img2, keypoints1, keypoints2, good, mask, homography if is_homography_valid else None)
        heatmap = self._heatmap(img1, keypoints1)
        ransac_vis = self._draw_ransac_comparison(
            img1, img2, keypoints1, keypoints2, good, mask,
            homography if is_homography_valid else None
        )

        t1 = time.perf_counter()
        return MatchResult(
            matched_vis,
            heatmap,
            homography if is_homography_valid else None,
            mask if is_homography_valid else None,
            similarity,
            similarity_good,
            inliers,
            reprojection_error,
            adaptive_threshold,
            stability,
            len(keypoints1),
            len(keypoints2),
            len(good),
            self.strategy.name,
            verdict,
            explanation,
            is_homography_valid,
            match_time_ms=(t1-t0)*1000,
            start_tick=t0,
            end_tick=t1,
            tick_diff=(t1-t0),
            ransac_vis=ransac_vis,
        )

    def _draw_ransac_comparison(self, img1, img2, kp1, kp2, good, mask, homography) -> np.ndarray:
        """Two-row visualization: all good matches (before) / inliers+outliers (after RANSAC)."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Row 1: before RANSAC — all good matches in yellow ──────────────
        before = cv2.drawMatches(
            img1, kp1, img2, kp2, good, None,
            matchColor=(0, 200, 220),
            singlePointColor=(120, 120, 120),
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        cv2.putText(before, f"До RANSAC: {len(good)} збігів (фільтр Lowe ratio)",
                    (10, 28), font, 0.7, (0, 200, 220), 2, cv2.LINE_AA)

        # ── Row 2: after RANSAC — inliers green, outliers red ──────────────
        if mask is not None:
            flat = mask.ravel()
            inlier_matches  = [m for m, k in zip(good, flat) if k]
            outlier_matches = [m for m, k in zip(good, flat) if not k]
            inlier_count  = len(inlier_matches)
            outlier_count = len(outlier_matches)
        else:
            inlier_matches, outlier_matches = good, []
            inlier_count, outlier_count = len(good), 0

        # draw inliers first
        after = cv2.drawMatches(
            img1, kp1, img2, kp2, inlier_matches, None,
            matchColor=(0, 220, 0),
            singlePointColor=(120, 120, 120),
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )
        # overlay outliers in red
        if outlier_matches:
            after = cv2.drawMatches(
                img1, kp1, img2, kp2, outlier_matches, after,
                matchColor=(0, 0, 220),
                singlePointColor=None,
                flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG,
            )
        # homography frame — validate before drawing
        if homography is not None and inlier_count >= 4:
            try:
                h, w = img1.shape[:2]
                corners = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(corners, homography)
                if proj is not None and np.all(np.isfinite(proj)):
                    max_dim = (img1.shape[1] + img2.shape[1]) * 3
                    if np.all(np.abs(proj) < max_dim):
                        proj_shifted = proj + np.array([w, 0], dtype=np.float32)
                        after = cv2.polylines(
                            after, [np.int32(proj_shifted)], True, (0, 255, 255), 3, cv2.LINE_AA
                        )
            except Exception:
                pass

        cv2.putText(after,
                    f"Після RANSAC: {inlier_count} інлайєрів (зелені)  |  {outlier_count} відкинуто (червоні)",
                    (10, 28), font, 0.7, (0, 220, 0), 2, cv2.LINE_AA)

        # ── Ensure same width before vstack ────────────────────────────────
        w_before, w_after = before.shape[1], after.shape[1]
        if w_before != w_after:
            target_w = max(w_before, w_after)
            def pad_w(img, w):
                pad = np.zeros((img.shape[0], w - img.shape[1], 3), dtype=np.uint8)
                return np.hstack([img, pad])
            if w_before < target_w:
                before = pad_w(before, target_w)
            else:
                after = pad_w(after, target_w)

        divider = np.full((6, before.shape[1], 3), (60, 60, 60), dtype=np.uint8)
        return np.vstack([before, divider, after])

    def _ratio_filter(self, matches: Sequence[Sequence[cv2.DMatch]]) -> list[cv2.DMatch]:
        good = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good.append(m)
        return good

    def _reprojection_error(self, src_pts, dst_pts, homography, mask) -> float:
        try:
            projected = cv2.perspectiveTransform(src_pts, homography)
            diffs = projected - dst_pts
            dists = np.linalg.norm(diffs.reshape(-1, 2), axis=1)
            inlier_dists = dists[mask.ravel() == 1]
            if inlier_dists.size == 0:
                return float(np.mean(dists)) if dists.size else 0.0
            return float(np.mean(inlier_dists))
        except Exception:
            return 0.0

    def _adaptive_threshold(self, similarity: float, similarity_good: float, good_count: int) -> float:
        # Adaptive threshold based on match statistics; stays within [5, 90]
        score = 0.6 * similarity + 0.3 * similarity_good + 0.1 * good_count
        return float(np.clip(score, 5.0, 90.0))

    def _heatmap(self, img: np.ndarray, keypoints) -> np.ndarray:
        if img is None or keypoints is None or len(keypoints) == 0:
            return img
        h, w = img.shape[:2]
        heat = np.zeros((h, w), dtype=np.float32)
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < w and 0 <= y < h:
                heat[y, x] += 1.0
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=10, sigmaY=10)
        cv2.normalize(heat, heat, 0, 255, cv2.NORM_MINMAX)
        heat_uint8 = heat.astype(np.uint8)
        colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
        return overlay

    def _draw_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        kp1,
        kp2,
        matches,
        mask,
        homography: Optional[np.ndarray],
    ) -> np.ndarray:
        if mask is not None:
            # Draw only RANSAC inliers in green
            matches_mask = mask.ravel().tolist()
            draw_params = dict(
                matchColor=(0, 220, 0),
                singlePointColor=(180, 180, 180),
                matchesMask=matches_mask,
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )
            vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
        else:
            # No valid mask — show top-30 best matches in blue-orange to avoid clutter
            top = sorted(matches, key=lambda m: m.distance)[:30]
            draw_params = dict(
                matchColor=(0, 165, 255),
                singlePointColor=(180, 180, 180),
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )
            vis = cv2.drawMatches(img1, kp1, img2, kp2, top, None, **draw_params)

        # Draw homography frame only when projection is numerically sane
        if homography is not None and mask is not None and mask.sum() >= 4:
            try:
                h, w = img1.shape[:2]
                corners = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                proj = cv2.perspectiveTransform(corners, homography)
                if proj is not None and np.all(np.isfinite(proj)):
                    max_dim = (img1.shape[1] + img2.shape[1]) * 3
                    if np.all(np.abs(proj) < max_dim):
                        proj_shifted = proj + np.array([w, 0], dtype=np.float32)
                        vis = cv2.polylines(
                            vis, [np.int32(proj_shifted)], True, (0, 255, 255), 3, cv2.LINE_AA
                        )
            except Exception:
                pass

        return vis

    def detect_and_match_sift_flann(self, img1: np.ndarray, img2: np.ndarray) -> MatchResult:
        """
        SIFT + FLANN matcher with automatic fallback to template matching if inliers < 8.
        Includes preprocessing: contrast enhancement, edge detection fallback.
        """
        import time
        t0 = time.perf_counter()
        gray1 = self._preprocess(img1)
        gray2 = self._preprocess(img2)

        # Try SIFT+FLANN first
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(gray1, None)
        kp2, desc2 = sift.detectAndCompute(gray2, None)

        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            t1 = time.perf_counter()
            res = self._template_matching_fallback(img1, img2, gray1, gray2)
            res.match_time_ms = (t1-t0)*1000
            return res

        # Use FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            knn_matches = flann.knnMatch(desc1, desc2, k=2)
        except Exception:
            t1 = time.perf_counter()
            res = self._template_matching_fallback(img1, img2, gray1, gray2)
            res.match_time_ms = (t1-t0)*1000
            return res

        # Lowe ratio test
        good = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < 4:
            t1 = time.perf_counter()
            res = self._template_matching_fallback(img1, img2, gray1, gray2)
            res.match_time_ms = (t1-t0)*1000
            return res

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        # Adaptive fallback: if < 8 inliers, use template matching
        if inliers < 8:
            t1 = time.perf_counter()
            res = self._template_matching_fallback(img1, img2, gray1, gray2)
            res.match_time_ms = (t1-t0)*1000
            return res

        min_kp = max(1, min(len(kp1), len(kp2)))
        similarity = 100.0 * inliers / min_kp
        similarity_good = 100.0 * len(good) / min_kp
        reprojection_error = self._reprojection_error(src_pts, dst_pts, homography, mask)
        stability = 1.0 / (1.0 + reprojection_error)

        adaptive_threshold = self._adaptive_threshold(similarity, similarity_good, len(good))
        matched_vis = self._draw_matches(img1, img2, kp1, kp2, good, mask, homography)
        heatmap = self._heatmap(img1, kp1)

        verdict = "🟢 Обʼєкт знайдено (SIFT)" if similarity >= 30 else "🟡 Часткова схожість (SIFT)"
        explanation = f"SIFT+FLANN: {inliers} інлайєрів, {similarity:.1f}% схожості"

        t1 = time.perf_counter()
        return MatchResult(
            matched_vis, heatmap, homography, mask,
            similarity, similarity_good, inliers, reprojection_error,
            adaptive_threshold, stability, len(kp1), len(kp2), len(good),
            "SIFT+FLANN", verdict, explanation, True,
            match_time_ms=(t1-t0)*1000,
            start_tick=t0,
            end_tick=t1,
            tick_diff=(t1-t0)
        )

    def _template_matching_fallback(self, img1: np.ndarray, img2: np.ndarray, 
                                    gray1: np.ndarray, gray2: np.ndarray) -> MatchResult:
        """
        Fallback to template matching (cv2.matchTemplate) when SIFT fails.
        Includes preprocessing: resize, contrast enhancement, edge detection.
        """
        # Resize template to match frame if needed
        h2, w2 = gray2.shape[:2]
        h1, w1 = gray1.shape[:2]
        
        # If template larger than frame, skip
        if h1 > h2 or w1 > w2:
            return MatchResult(
                None, None, None, None, 0.0, 0.0, 0, 0.0, 0.0, 0.0,
                h1, h2, 0, "Template", "🔴 Обʼєкт не знайдено",
                "Шаблон більший за кадр", False
            )

        # Try multiple preprocessing variants
        best_score = 0.0
        best_method = "direct"

        # Variant 1: Direct template matching on grayscale
        result = cv2.matchTemplate(gray2, gray1, cv2.TM_CCOEFF_NORMED)
        score1 = np.max(result) if result.size > 0 else 0.0

        # Variant 2: Contrast-enhanced (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray1_clahe = clahe.apply(gray1)
        gray2_clahe = clahe.apply(gray2)
        result = cv2.matchTemplate(gray2_clahe, gray1_clahe, cv2.TM_CCOEFF_NORMED)
        score2 = np.max(result) if result.size > 0 else 0.0

        # Variant 3: Edge detection (Canny)
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        if edges1.sum() > 0 and edges2.sum() > 0:
            result = cv2.matchTemplate(edges2, edges1, cv2.TM_CCOEFF_NORMED)
            score3 = np.max(result) if result.size > 0 else 0.0
        else:
            score3 = 0.0

        # Pick best variant
        if score2 >= score1 and score2 >= score3:
            best_score = score2
            best_method = "CLAHE"
        elif score3 > score1:
            best_score = score3
            best_method = "Canny"
        else:
            best_score = score1
            best_method = "direct"

        # Convert correlation to percentage similarity
        similarity = max(0.0, best_score * 100.0)

        # Create visualization
        if best_score > 0.3:  # Threshold for acceptable match
            result = cv2.matchTemplate(gray2_clahe if best_method == "CLAHE" else (edges2 if best_method == "Canny" else gray2),
                                      gray1_clahe if best_method == "CLAHE" else (edges1 if best_method == "Canny" else gray1),
                                      cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc
            bottom_right = (top_left[0] + w1, top_left[1] + h1)
            
            matched_vis = img2.copy()
            cv2.rectangle(matched_vis, top_left, bottom_right, (0, 255, 255), 2)
        else:
            matched_vis = None

        # Heatmap from gray2
        heatmap = self._heatmap(img2, [])

        verdict = "🟢 Обʼєкт знайдено (Template)" if similarity >= 30 else "🟡 Часткова схожість (Template)"
        explanation = f"Template matching ({best_method}): {similarity:.1f}% схожості"

        t1 = time.perf_counter()
        return MatchResult(
            matched_vis, heatmap, None, None,
            similarity, similarity, 0, 0.0,
            similarity, 1.0 if similarity > 30 else 0.5, h1, h2, 0,
            f"Template({best_method})", verdict, explanation, best_score > 0.3,
            match_time_ms=(t1-t0)*1000,
            start_tick=t0,
            end_tick=t1,
            tick_diff=(t1-t0)
        )
