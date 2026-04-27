from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import cv2


class FeatureStrategy(ABC):
    """Strategy for feature detector/descriptor and matcher config."""

    name: str = "base"
    norm: int = cv2.NORM_HAMMING

    @abstractmethod
    def create_detector(self):
        raise NotImplementedError

    def create_matcher(self, cross_check: bool = False):
        # BFMatcher is enough for the required algorithms
        return cv2.BFMatcher(self.norm, crossCheck=cross_check)


class ORBStrategy(FeatureStrategy):
    name = "ORB"
    norm = cv2.NORM_HAMMING

    def __init__(self, nfeatures: int = 5000):
        self.nfeatures = nfeatures

    def create_detector(self):
        return cv2.ORB_create(nfeatures=self.nfeatures)


class AKAZEStrategy(FeatureStrategy):
    name = "AKAZE"
    norm = cv2.NORM_HAMMING

    def __init__(self):
        self.descriptor_type = cv2.AKAZE_DESCRIPTOR_MLDB

    def create_detector(self):
        return cv2.AKAZE_create(descriptor_type=self.descriptor_type)


class SIFTStrategy(FeatureStrategy):
    name = "SIFT"
    norm = cv2.NORM_L2

    def __init__(self, nfeatures: int = 0):
        self.nfeatures = nfeatures

    def create_detector(self):
        # SIFT is available in recent opencv-python builds; guard in UI
        return cv2.SIFT_create(nfeatures=self.nfeatures)


def get_available_strategies() -> list[FeatureStrategy]:
    strategies: list[FeatureStrategy] = [ORBStrategy(), AKAZEStrategy()]
    # Check if SIFT is available in current OpenCV build
    if hasattr(cv2, "SIFT_create"):
        try:
            strategies.append(SIFTStrategy())
        except Exception:
            pass
    return strategies
