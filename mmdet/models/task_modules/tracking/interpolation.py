# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor as GPR
    from sklearn.gaussian_process.kernels import RBF
    HAS_SKIKIT_LEARN = True
except ImportError:
    HAS_SKIKIT_LEARN = False

from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class InterpolateTracklets:
    """Interpolate tracks to make tracks more complete.

    Args:
        min_num_frames (int, optional): The minimum length of a track that will
            be interpolated. Defaults to 5.
        max_num_frames (int, optional): The maximum disconnected length in
            a track. Defaults to 20.
        use_gsi (bool, optional): Whether to use the GSI (Gaussian-smoothed
            interpolation) method. Defaults to False.
        smooth_tau (int, optional): smoothing parameter in GSI. Defaults to 10.
    """

    def __init__(self,
                 min_num_frames: int = 5,
                 max_num_frames: int = 20,
                 use_gsi: bool = False,
                 smooth_tau: int = 10):
        if not HAS_SKIKIT_LEARN:
            raise RuntimeError('sscikit-learn is not installed,\
                 please install it by: pip install scikit-learn')
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.use_gsi = use_gsi
        self.smooth_tau = smooth_tau

    def _interpolate_track(self,
                           track: np.ndarray,
                           track_id: int,
                           max_num_frames: int = 20) -> np.ndarray:
        """Interpolate a track linearly to make the track more complete.

        This function is proposed in
        "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
        `ByteTrack<https://arxiv.org/abs/2110.06864>`_.

        Args:
            track (ndarray): With shape (N, 7). Each row denotes
                (frame_id, track_id, x1, y1, x2, y2, score).
            max_num_frames (int, optional): The maximum disconnected length in
                the track. Defaults to 20.

        Returns:
            ndarray: The interpolated track with shape (N, 7). Each row denotes
                (frame_id, track_id, x1, y1, x2, y2, score)
        """
        assert (track[:, 1] == track_id).all(), \
            'The track id should not changed when interpolate a track.'

        frame_ids = track[:, 0]
        interpolated_track = np.zeros((0, 7))
        # perform interpolation for the disconnected frames in the track.
        for i in np.where(np.diff(frame_ids) > 1)[0]:
            left_frame_id = frame_ids[i]
            right_frame_id = frame_ids[i + 1]
            num_disconnected_frames = int(right_frame_id - left_frame_id)

            if 1 < num_disconnected_frames < max_num_frames:
                left_bbox = track[i, 2:6]
                right_bbox = track[i + 1, 2:6]

                # perform interpolation for two adjacent tracklets.
                for j in range(1, num_disconnected_frames):
                    cur_bbox = j / (num_disconnected_frames) * (
                        right_bbox - left_bbox) + left_bbox
                    cur_result = np.ones((7, ))
                    cur_result[0] = j + left_frame_id
                    cur_result[1] = track_id
                    cur_result[2:6] = cur_bbox

                    interpolated_track = np.concatenate(
                        (interpolated_track, cur_result[None]), axis=0)

        interpolated_track = np.concatenate((track, interpolated_track),
                                            axis=0)
        return interpolated_track

    def gaussian_smoothed_interpolation(self,
                                        track: np.ndarray,
                                        smooth_tau: int = 10) -> np.ndarray:
        """Gaussian-Smoothed Interpolation.

        This function is proposed in
        "StrongSORT: Make DeepSORT Great Again"
        `StrongSORT<https://arxiv.org/abs/2202.13514>`_.

        Args:
            track (ndarray): With shape (N, 7). Each row denotes
                (frame_id, track_id, x1, y1, x2, y2, score).
            smooth_tau (int, optional): smoothing parameter in GSI.
                Defaults to 10.

        Returns:
            ndarray: The interpolated tracks with shape (N, 7). Each row
                denotes (frame_id, track_id, x1, y1, x2, y2, score)
        """
        len_scale = np.clip(smooth_tau * np.log(smooth_tau**3 / len(track)),
                            smooth_tau**-1, smooth_tau**2)
        gpr = GPR(RBF(len_scale, 'fixed'))
        t = track[:, 0].reshape(-1, 1)
        x1 = track[:, 2].reshape(-1, 1)
        y1 = track[:, 3].reshape(-1, 1)
        x2 = track[:, 4].reshape(-1, 1)
        y2 = track[:, 5].reshape(-1, 1)
        gpr.fit(t, x1)
        x1_gpr = gpr.predict(t)
        gpr.fit(t, y1)
        y1_gpr = gpr.predict(t)
        gpr.fit(t, x2)
        x2_gpr = gpr.predict(t)
        gpr.fit(t, y2)
        y2_gpr = gpr.predict(t)
        gsi_track = [[
            t[i, 0], track[i, 1], x1_gpr[i], y1_gpr[i], x2_gpr[i], y2_gpr[i],
            track[i, 6]
        ] for i in range(len(t))]
        return np.array(gsi_track)

    def forward(self, pred_tracks: np.ndarray) -> np.ndarray:
        """Forward function.

        pred_tracks (ndarray): With shape (N, 7). Each row denotes
            (frame_id, track_id, x1, y1, x2, y2, score).

        Returns:
            ndarray: The interpolated tracks with shape (N, 7). Each row
            denotes (frame_id, track_id, x1, y1, x2, y2, score).
        """
        max_track_id = int(np.max(pred_tracks[:, 1]))
        min_track_id = int(np.min(pred_tracks[:, 1]))

        # perform interpolation for each track
        interpolated_tracks = []
        for track_id in range(min_track_id, max_track_id + 1):
            inds = pred_tracks[:, 1] == track_id
            track = pred_tracks[inds]
            num_frames = len(track)
            if num_frames <= 2:
                continue

            if num_frames > self.min_num_frames:
                interpolated_track = self._interpolate_track(
                    track, track_id, self.max_num_frames)
            else:
                interpolated_track = track

            if self.use_gsi:
                interpolated_track = self.gaussian_smoothed_interpolation(
                    interpolated_track, self.smooth_tau)

            interpolated_tracks.append(interpolated_track)

        interpolated_tracks = np.concatenate(interpolated_tracks)
        return interpolated_tracks[interpolated_tracks[:, 0].argsort()]
