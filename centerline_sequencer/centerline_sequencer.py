from typing import Optional
import numpy as np
from numpy import ndarray

import torch
from torch_cluster import fps


def compute_min_nn_dist(points: ndarray):
    """
    Computes distance for each point to its nearest neighbor.

    Arguments:
        points: ndarray of shape `(N, D)`.

    Returns:
        Array of shape `(N,)` of minimum distances.
    """
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    dists[np.diag_indices_from(dists)] = np.inf
    min_nn_dists = np.min(dists, axis=-1)

    return min_nn_dists


def remove_outliers(
    points: ndarray, features: ndarray, mult: float = 4.0, reverse: bool = False
) -> tuple[ndarray, ndarray]:
    """
    Removes points that are more than `mult` times the average nearest
    neighbor distance away from their nearest neighbor.

    Arguments:
        points: ndarray of shape `(N, D)` of points.
        features: ndarray of shape `(N, F)` of point features.
        mult: float specifying the average nearest neighbor distance
              multiplyer.
        reverse: if True, will keep outliers instead of removing them.

    Returns:
        Points and features with outliers removed.
    """
    min_nn_dists = compute_min_nn_dist(points)
    avg_min_nn_dist = min_nn_dists.mean()

    keep = min_nn_dists <= (mult * avg_min_nn_dist)

    keep = keep if reverse is False else ~keep

    return points[keep], features[keep]


def downsample(
    points: ndarray,
    features: ndarray,
    ratio: float = 0.5,
    random_start: bool = True,
    preserve_order: bool = False,
) -> tuple[ndarray, ndarray]:
    """
    Downsamples given points using furthest point sampling with given ratio.
    Preserves point cloud structure but reduces point clustering.

    Arguments:
        - points: ndarray of shape `(N, D)` of points.
        - features: ndarray of shape `(N, F)` of point features.
        - ratio: float specifies the down sample ratio. Downsampled point
                 cloud is of (approximate) shape `(ratio * N, ...)`.
        - random_start: selects random starting point for fps sampling if True.
                        If False, first element in points will be used.
        - preserve_order: If True, preserves input ordering.

    Returns:
        Downsampled points and features.
    """
    points, features = map(torch.from_numpy, (points, features))

    idc = fps(points, ratio=ratio, random_start=random_start)

    if preserve_order:
        idc = np.sort(idc)

    points, features = map(lambda x: x.numpy(), (points[idc], features[idc]))

    return points, features


def compute_start_point_idx(points: ndarray, k: int = 5) -> int:
    """
    Computes the most likely starting point of an unordered set of points
    that form a line by pairwise relative inner products of the `k` nearest
    neighbors.

    Arguments:
        - points: ndarray of shape `(N, D)` of points.
        - k: number of nearest neighbors to consider.

    Returns:
        Index of most likely starting point.
    """
    dists = np.linalg.norm(points[:, None] - points, axis=-1)
    dists[np.diag_indices_from(dists)] = np.inf
    idc = np.argpartition(dists, k, axis=-1)[..., :k]

    neighbors = points[idc]
    orientations = points[:, None] - neighbors
    orientations /= np.linalg.norm(orientations, axis=-1, keepdims=True)

    # pairwise inner product
    dirs = (orientations @ orientations.transpose(0, -1, -2)).reshape(-1, k * k)
    mean_dirs = dirs.mean(-1)

    start_points_idx = np.argmax(mean_dirs)

    return start_points_idx


def compute_nearest_neighbor(
    point: ndarray, neighbors: ndarray, weight: float | ndarray = 1
) -> tuple[ndarray, ndarray]:
    """
    Computes the nearest neighbor of point from neighbors.

    Arguments:
        - point: ndarray of shape `(D,)` of points.
        - neighbors: ndarray of shape `(N, D)` of points.
        - weight: multiplicative factor applied to the nearest neighbor
                  distances. If ndarray, shape should be `(N,)`.

    Returns:
        Index of minimum neighbor and corresponding distance.
    """
    dists = weight * np.linalg.norm(point[None] - neighbors, axis=-1)
    idx = np.argmin(dists)

    return idx, dists[idx]


def compute_flow_weights(
    point: ndarray, neighbors: ndarray, flow: ndarray, weight: float = 0.25
) -> ndarray:
    """
    The flow weight is the inverse similarity from the current point flow to the flow
    from point to all neighbors. As flow weight increases, the corresponding neighbor
    moves more against the direction of the current flow from fiven point.

    Flows are normalized.

    Arguments:
        - point: ndarray of shape `(D,)` for which to compute flow weights for.
        - neighbors: ndarray of shape `(N, D)` of neighboring points.
        - flow: ndarray of shape `(D,)` of current flow.
        - weight: weight given to flow similarity. Weights are centered around 1, that is
                  flow weights are in the range of [1 - weight, 1 + weight].

    returns:
        Array of shape `(N,)` of flow weights.
    """
    flow /= np.linalg.norm(flow)

    flows = neighbors - point
    flows /= np.linalg.norm(flows, axis=-1, keepdims=True)

    flow_weights = -weight * (flow @ flows.T) + 1

    return flow_weights


def merge_points(
    points: ndarray,
    features: ndarray,
    flow: bool = True,
    min_segment_length: int = 5,
    flow_weight: float = 0.25,
) -> tuple[ndarray, ndarray]:
    """
    Merges unconnected points to form a line.

    Arguments:
        points: ndarray of shape `(N, D)` of points.
        features: ndarray of shape `(N, F)` of point features.
        flow: if True, includes line flow information to connect points.

    Returns:
        Points and features sorted such that they form a single line.
    """
    unconnected = list(i for i in range(0, points.shape[0]))

    start_idx = compute_start_point_idx(points)
    connected = [start_idx]
    unconnected.pop(start_idx)

    cma = 0
    current_flow = None
    for i in range(len(unconnected)):
        current = connected[-1]

        flow_weights = (
            1
            if not flow or current_flow is None
            else compute_flow_weights(
                points[current],
                points[unconnected],
                flow=current_flow,
                weight=flow_weight,
            )
        )

        nn, dist = compute_nearest_neighbor(
            points[current], points[unconnected], weight=flow_weights
        )
        nn = unconnected[nn]

        if i != 0 and dist > min_segment_length * cma:
            break

        current_flow = points[nn] - points[current]

        connected.append(nn)
        unconnected.remove(nn)

        # cumulative moving average
        cma = (i * cma + dist) / (i + 1)

    return points[connected], features[connected]


def merge_segments(
    segments: list[ndarray], segment_features: list[ndarray], mult: float = 4.0
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Merges different line segments to form a tree of line segments. Assumes that two
    segments that are merged are connected through at least one end point.

    Arguments:
        segments: list containing ndarrays of points.
        segment_features: list containing ndarrays of point features.
        mult: multiplicative threshold for determining whether two points from different
              segments should be the merging point.

    Returns:
        Lists of segments and features with now overlapping points denoting their
        merging point.
    """
    endpoint_idc = [0, -1]

    for t, (segment, features) in enumerate(zip(segments, segment_features)):
        endpoints = segment[endpoint_idc]
        endpoints_features = features[endpoint_idc]

        avg_min_dist = compute_min_nn_dist(segment).mean()

        for i, (endpoint, endpoints_feature) in enumerate(
            zip(endpoints, endpoints_features)
        ):
            ts, nns, candidate_points, candidate_features = [], [], [], []

            for c_t, (candidate_segment, candidate_feature) in enumerate(
                zip(segments, segment_features)
            ):
                if t == c_t:
                    continue  # no need to connect a segment to itself

                nn, dist = compute_nearest_neighbor(endpoint, candidate_segment)

                if (dist > mult * avg_min_dist) or (dist < 1e-7):
                    continue  # too far away or already connected

                # indices of candidate segment points we want to merge
                ts.append(c_t)
                nns.append(nn)
                candidate_points.append(candidate_segment[nn])
                candidate_features.append(candidate_feature[nn])

            if not ts:
                continue  # in the case of an empty segment

            ts.append(t)
            nns.append(-1 * i)
            candidate_points.append(endpoint)
            candidate_features.append(endpoints_feature)

            merged_point = np.asarray(candidate_points).mean(0)
            merged_features = np.asarray(candidate_features).mean(0)

            for c_t, nn in zip(ts, nns):
                segments[c_t][nn] = merged_point
                segment_features[c_t][nn] = merged_features

    return segments, segment_features


def merge_segments_with_connections(
    segments: list[ndarray],
    segment_features: list[ndarray],
    connections=list[list[int]],
) -> tuple[list[ndarray], list[ndarray]]:
    for source, targets in connections:
        pass


def merge_centerline_segmented(
    points: ndarray,
    features: ndarray,
    segment_types: ndarray,
    min_segment_length: int = 5,
    flow_weight: float = 0.25,
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Merges unsorted centerline points into a connected centerline composed
    of different line segments based on segment types.

    Arguments:
        points: ndarray of shape `(N, D)` of points.
        features: ndarray of shape `(N, F)` of point features. Features should
                  be continuous as they can be interpolated during the merging
                  process.
        segment_types: ndarray of shape `(N,)` of the segment each point
                       belongs to.

    Returns:
        Lists containing merged segments of points and features, corresponding to
        the given segment types.
    """
    types = np.unique(segment_types)

    features = np.concatenate((features, segment_types[..., None]), axis=-1)

    points, features = remove_outliers(points, features, 4)
    points, features = downsample(points, features, 0.66666)

    features, segment_types = features[..., :-1], features[..., -1]

    merged_centerline = []
    merged_features = []

    for t in types:
        segmentation = segment_types == t
        points_t, features_t = points[segmentation], features[segmentation]

        if points_t.shape[0] < min_segment_length:
            continue

        points_t, features_t = merge_points(
            points_t,
            features_t,
            flow=True,
            min_segment_length=min_segment_length,
            flow_weight=flow_weight,
        )
        points_t, features_t = downsample(
            points_t, features_t, 0.666666, random_start=False, preserve_order=True
        )

        merged_centerline.append(points_t)
        merged_features.append(features_t)

    centerline, features = merge_segments(merged_centerline, merged_features)

    return centerline, features


def merge_centerline_unsegmented(
    points: ndarray, features: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Merges unsorted centerline points into a connected centerline composed
    of different line segments.

    Arguments:
        points: ndarray of shape `(N, D)` of points.
        features: ndarray of shape `(N, F)` of point features. Features should
                  be continuous as they can be interpolated during the merging
                  process.

    Returns:
        Lists containing merged segments of points and features, corresponding to
        the given segment types.
    """
    D = points.shape[-1]

    merged_centerline = []
    merged_features = []

    points, features = remove_outliers(points, features, 4)
    # points, features = downsample(points, features, 0.6666666666)

    buffer = set((*point, *feature) for point, feature in zip(points, features))

    while buffer:
        points_t, features_t = merge_points(points, features)

        # buffer update
        for point, feature in zip(points_t, features_t):
            buffer.remove((*point, *feature))

        points_t, features_t = downsample(
            points_t, features_t, 0.6666666666, random_start=False, preserve_order=True
        )

        # segments that are very short are probably noise
        if len(points_t) < 5:
            continue

        merged_centerline.append(points_t)
        merged_features.append(features_t)

        if len(buffer) < 5:
            continue

        buffer_array = np.asarray(list(buffer))
        points, features = buffer_array[:, :D], buffer_array[:, D:]

    centerline, features = merge_segments(merged_centerline, merged_features)

    return centerline, features


def merge_centerline(
    points: ndarray,
    features: ndarray,
    segment_types: Optional[ndarray] = None,
    min_segment_length: int = 5,
    flow_weight: float = 0.25,
) -> tuple[list[ndarray], list[ndarray]]:
    """
    Merges unsorted centerline points into a connected centerline composed
    of different line segments based on segment types.

    Arguments:
        points: ndarray of shape `(N, D)` of points.
        features: ndarray of shape `(N, F)` of point features. Features should
                  be continuous as they can be interpolated during the merging
                  process.
        segment_types: Optional integer ndarray of shape `(N,)` of the segment each
                       point belongs to. If provided, will create centerline segments
                       corresponding to each segment type.

    Returns:
        Lists containing merged segments of points and features, corresponding to
        the given segment types.
    """
    N = points.shape[0]
    features = features.reshape(N, -1)

    if segment_types is None:
        return merge_centerline_unsegmented(points, features)

    return merge_centerline_segmented(
        points, features, segment_types, min_segment_length, flow_weight
    )
