from typing import Generator, Tuple
import time
import cv2
import numpy as np

from .matcher import ImageMatcher, MatchResult


class VideoProcessor:
    """Process video frames to find matching objects using SIFT+FLANN with template matching fallback."""
    
    def __init__(self, frame_skip: int = 10, similarity_threshold: float = 10.0):
        self.frame_skip = max(1, frame_skip)
        self.similarity_threshold = similarity_threshold
        self.paused = False
        self.stopped = False
        self.use_sift = True  # New: enable/disable SIFT+FLANN

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stopped = True

    def toggle_pause(self):
        self.paused = not self.paused

    def process_video(
        self,
        video_path: str,
        reference_image: np.ndarray,
        matcher: ImageMatcher,
    ) -> Generator[Tuple[int, int, MatchResult], None, None]:
        """
        Process video frames and yield matching frames.
        Uses SIFT+FLANN if available, automatically falls back to template matching.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ ERROR: Cannot open video: {video_path}")
            return

        # Get video properties for diagnostics
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video info:")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps:.1f}")
        print(f"   Resolution: {width}x{height}")
        print(f"   Duration: {total_frames/fps:.1f} sec" if fps > 0 else "   Duration: unknown")
        print(f"   Frame skip: {self.frame_skip} (analyzing every {self.frame_skip} frames)")
        print(f"   Similarity threshold: {self.similarity_threshold}%")
        print(f"   Reference image size: {reference_image.shape[1]}x{reference_image.shape[0]}")
        
        frame_idx = 0
        matched_count = 0
        processed_count = 0
        
        try:
            while True:
                if self.stopped:
                    break
                    
                if self.paused:
                    time.sleep(0.05)
                    continue

                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_idx % self.frame_skip != 0:
                    frame_idx += 1
                    continue

                processed_count += 1
                print(f"\n🔍 Processing frame {frame_idx}/{total_frames} (#{processed_count} analyzed)")

                # Use SIFT+FLANN with template matching fallback
                import time
                t0 = time.perf_counter()
                method_used = "unknown"
                if self.use_sift and hasattr(cv2, 'SIFT_create'):
                    try:
                        match_result = matcher.detect_and_match_sift_flann(reference_image, frame)
                        method_used = "SIFT+FLANN"
                    except Exception as e:
                        print(f"   ⚠️ SIFT failed: {e}, falling back to {matcher.strategy.name}")
                        match_result = matcher.detect_and_match(reference_image, frame)
                        method_used = matcher.strategy.name
                else:
                    match_result = matcher.detect_and_match(reference_image, frame)
                    method_used = matcher.strategy.name
                # t0, t1, and tick_diff are now in match_result

                # Dynamic threshold: max(user_threshold, adaptive)
                # Fix: Use user threshold when it's lower than adaptive (adaptive can be too strict)
                dynamic_threshold = min(self.similarity_threshold, match_result.adaptive_threshold) if match_result.adaptive_threshold > 0 else self.similarity_threshold
                
                print(f"   Method: {method_used} | Actual: {match_result.method}")
                print(f"   Keypoints: ref={match_result.keypoints1}, frame={match_result.keypoints2}")
                print(f"   Good matches: {match_result.good_matches}")
                print(f"   Inliers: {match_result.inliers}")
                print(f"   Similarity: {match_result.similarity:.1f}% (threshold: {dynamic_threshold:.1f}%)")
                print(f"   Homography: {'✓ valid' if match_result.homography is not None else '✗ none'}")
                print(f"   ⏱ Час метчінгу: {match_result.match_time_ms:.1f} мс | Тіки: start={match_result.start_tick:.6f}, end={match_result.end_tick:.6f}, Δ={match_result.tick_diff:.6f}")
                
                # Accept if: valid homography + sufficient similarity OR template match > 30%
                is_valid = (
                    (match_result.homography is not None and match_result.similarity >= dynamic_threshold) or
                    (match_result.homography is None and match_result.similarity >= 25.0)  # Lower threshold for template
                )
                
                if is_valid:
                    matched_count += 1
                    print(f"   ✅ MATCH ACCEPTED (total matches: {matched_count})")
                    yield frame_idx, total_frames, match_result
                else:
                    print(f"   ❌ Frame rejected")

                frame_idx += 1
                
        finally:
            cap.release()
            print(f"\n📊 Video processing complete:")
            print(f"   Total frames: {total_frames}")
            print(f"   Frames analyzed: {processed_count}")
            print(f"   Frames matched: {matched_count}")
            if processed_count > 0:
                print(f"   Match rate: {matched_count/processed_count*100:.1f}%")
            if matched_count == 0:
                print(f"\n⚠️ WARNING: No frames matched!")
                print(f"   Possible reasons:")
                print(f"   • Reference image doesn't match video content")
                print(f"   • Video quality too low (try original MOV file)")
                print(f"   • Threshold too high (current: {self.similarity_threshold}%)")
                print(f"   • Frame skip too high (current: {self.frame_skip})")

