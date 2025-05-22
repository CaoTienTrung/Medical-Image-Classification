import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
import cv2


class HOGFeatureExtractor:
    def __init__(self):
        self.params = {
            "orientations": 9,
            "pixels_per_cell": (8, 8),
            "cells_per_block": (2, 2),
            "block_norm": "L2-Hys",
            "feature_vector": True,
        }

    def extract(self, img):
        return hog(
            img,
            orientations=self.params["orientations"],
            pixels_per_cell=self.params["pixels_per_cell"],
            cells_per_block=self.params["cells_per_block"],
            block_norm=self.params["block_norm"],
            feature_vector=self.params["feature_vector"],
        )


class LBPFeatureExtractor:
    def __init__(self):
        self.params = {"P": 8, "R": 1, "method": "uniform"}

    def extract(self, img):
        lbp = local_binary_pattern(
            img, P=self.params["P"], R=self.params["R"], method=self.params["method"]
        )
        nbins = self.params["P"] + 2
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, nbins + 1), range=(0, nbins), density=True
        )
        return hist


class GLCMFeatureExtractor:
    def __init__(self):
        self.distances = [1]
        self.angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.properties = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]

    def extract(self, img):
        glcm = graycomatrix(
            img,
            distances=self.distances,
            angles=self.angles,
            symmetric=True,
            normed=True,
        )

        return np.concatenate([graycoprops(glcm, p).flatten() for p in self.properties])


class GaborExtractor:
    def __init__(self):
        self.params = {
            "ksize": (21, 21),
            "sigmas": [1, 3],
            "thetas": np.linspace(0, np.pi, 4, endpoint=False),
            "lambdas": [np.pi / 4, np.pi / 2],
            "gamma": 0.5,
            "psi": 0,
        }

    def extract(self, image):
        p = self.params
        feats = []
        image = image.astype(np.float32)
        for sigma in p["sigmas"]:
            for theta in p["thetas"]:
                for lam in p["lambdas"]:
                    kernel = cv2.getGaborKernel(
                        ksize=p["ksize"],
                        sigma=sigma,
                        theta=theta,
                        lambd=lam,
                        gamma=p["gamma"],
                        psi=p["psi"],
                        ktype=cv2.CV_32F,
                    )
                    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                    feats.append(filtered.mean())
                    feats.append(filtered.var())
        return np.array(feats)


class SIFTFeatureExtractor:
    def __init__(self, max_keypoints=500):
        """
        max_keypoints : int
           Số keypoint lớn nhất sẽ giữ lại; nếu ít hơn sẽ được padding zeros.
        """
        self.sift = cv2.SIFT_create()
        self.max_keypoints = max_keypoints

    def extract(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        # Nếu không có descriptor, tạo mảng rỗng shape=(0,128)
        if descriptors is None:
            descriptors = np.zeros((0, 128), dtype=np.float32)

        # Sắp xếp theo response strength (ưu tiên keypoint tin cậy nhất)
        if keypoints:
            responses = np.array([kp.response for kp in keypoints])
            order = np.argsort(-responses)
            descriptors = descriptors[order]

        # Pad hoặc truncate về đúng số max_keypoints
        n = descriptors.shape[0]
        if n < self.max_keypoints:
            padding = np.zeros((self.max_keypoints - n, 128), dtype=descriptors.dtype)
            descriptors = np.vstack([descriptors, padding])
        else:
            descriptors = descriptors[: self.max_keypoints]
        return descriptors.flatten()
