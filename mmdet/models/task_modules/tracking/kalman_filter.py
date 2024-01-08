# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import numpy as np
import torch

try:
    import scipy.linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class KalmanFilter:
    """A simple Kalman filter for tracking bounding boxes in image space.

    The implementation is referred to https://github.com/nwojke/deep_sort.

    Args:
        center_only (bool): If True, distance computation is done with
            respect to the bounding box center position only.
            Defaults to False.
        use_nsa (bool): Whether to use the NSA (Noise Scale Adaptive) Kalman
            Filter, which adaptively modulates the noise scale according to
            the quality of detections. More details in
            https://arxiv.org/abs/2202.11983. Defaults to False.
    """
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def __init__(self, center_only: bool = False, use_nsa: bool = False):
        if not HAS_SCIPY:
            raise RuntimeError('sscikit-learn is not installed,\
                 please install it by: pip install scikit-learn')
        self.center_only = center_only
        if self.center_only:
            self.gating_threshold = self.chi2inv95[2]
        else:
            self.gating_threshold = self.chi2inv95[4]

        self.use_nsa = use_nsa
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.array) -> Tuple[np.array, np.array]:
        """Create track from unassociated measurement.

        Args:
            measurement (ndarray):  Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.

        Returns:
             (ndarray, ndarray): Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3], 1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.array,
                covariance: np.array) -> Tuple[np.array, np.array]:
        """Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.

            covariance (ndarray): The 8x8 dimensional covariance matrix
                of the object state at the previous time step.

        Returns:
            (ndarray, ndarray): Returns the mean vector and covariance
                matrix of the predicted state. Unobserved velocities are
                initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self,
                mean: np.array,
                covariance: np.array,
                bbox_score: float = 0.) -> Tuple[np.array, np.array]:
        """Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            bbox_score (float): The confidence score of the bbox.
                Defaults to 0.

        Returns:
            (ndarray, ndarray):  Returns the projected mean and covariance
            matrix of the given state estimate.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-1,
            self._std_weight_position * mean[3]
        ]

        if self.use_nsa:
            std = [(1 - bbox_score) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self,
               mean: np.array,
               covariance: np.array,
               measurement: np.array,
               bbox_score: float = 0.) -> Tuple[np.array, np.array]:
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the
                aspect ratio, and h the height of the bounding box.
            bbox_score (float): The confidence score of the bbox.
                Defaults to 0.

        Returns:
             (ndarray, ndarray): Returns the measurement-corrected state
             distribution.
        """
        projected_mean, projected_cov = \
            self.project(mean, covariance, bbox_score)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance,
                                                    self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self,
                        mean: np.array,
                        covariance: np.array,
                        measurements: np.array,
                        only_position: bool = False) -> np.array:
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.
            only_position (bool, optional): If True, distance computation is
                done with respect to the bounding box center position only.
                Defaults to False.

        Returns:
            ndarray: Returns an array of length N, where the i-th element
            contains the squared Mahalanobis distance between
            (mean, covariance) and `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

    def track(self, tracks: dict,
              bboxes: torch.Tensor) -> Tuple[dict, np.array]:
        """Track forward.

        Args:
            tracks (dict[int:dict]): Track buffer.
            bboxes (Tensor): Detected bounding boxes.

        Returns:
            (dict[int:dict], ndarray): Updated tracks and bboxes.
        """
        costs = []
        for id, track in tracks.items():
            track.mean, track.covariance = self.predict(
                track.mean, track.covariance)
            gating_distance = self.gating_distance(track.mean,
                                                   track.covariance,
                                                   bboxes.cpu().numpy(),
                                                   self.center_only)
            costs.append(gating_distance)

        costs = np.stack(costs, 0)
        costs[costs > self.gating_threshold] = np.nan
        return tracks, costs
