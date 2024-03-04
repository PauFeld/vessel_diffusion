import os
from git import Optional
import numpy as np

from matplotlib import pyplot as plt


def smooth_curve(intervals, start_value=1, half_curve=True):
    start = 0
    curve = []

    for i, (end, end_value) in enumerate(intervals):
        half_curve = half_curve and (i == 0 or i == len(intervals) - 1)

        l = end - start

        x = np.linspace(0, l, l, endpoint=False)

        if half_curve:
            curve.append(
                (end_value - start_value) * np.sin((np.pi * x) / (2 * (l)))
                + start_value
            )
        else:
            dab = (end_value - start_value) / 2
            curve.append(dab * np.cos((np.pi * (x - l)) / l) + dab + start_value)

        start = end
        start_value = end_value

    return np.concatenate(curve, axis=0)


def rotation_matrix(alpha, beta, gamma):
    cosa, cosb, cosg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sina, sinb, sing = np.sin(alpha), np.sin(beta), np.sin(gamma)

    return np.array(
        [
            [
                cosa * cosb,
                cosa * sinb * sing - sina * cosg,
                cosa * sinb * cosg + sina * sing,
            ],
            [
                sina * cosb,
                sina * sinb * sing + cosa * cosg,
                sina * sinb * cosg - cosa * sing,
            ],
            [-sinb, cosb * sing, cosb * cosg],
        ]
    )


def generate_vessel_segment(
    num_points: int = 256,
    length: float = 30,
    origin: tuple[int, ...] = (0, 0, 0),
    initial_direction: tuple = (0.3, 0.3, 1),
    tortuosity: float = 0.6,
    curve_bias: tuple = (0.5, 0.01, 0.005),
    radius_init: float = 3,
    radius_intervals: Optional[list[tuple[int, float]]] = None,
):
    radii = (
        radius_init + np.zeros(num_points)
        if radius_intervals is None
        else smooth_curve(radius_intervals, start_value=radius_init)
    )
    radii += 0.01 * radii.max() * np.random.randn(num_points)

    segment = [np.asarray(origin[:3])]

    step_size = length / num_points

    curve_bias = np.asarray(curve_bias)

    current_direction = np.asarray(initial_direction)
    current_direction /= np.linalg.norm(current_direction)
    current_direction *= step_size

    for _ in range(num_points - 1):
        point = segment[-1]

        next_point = point + current_direction

        segment.append(next_point)

        angles = 0.1 * tortuosity * (curve_bias + np.random.randn(3))
        rotation = rotation_matrix(*angles)

        current_direction = current_direction @ rotation.T

    segment = np.asarray(segment)
    radii = np.asarray(radii).reshape(-1, 1)

    segment = np.vstack((np.concatenate((segment, radii), axis=-1)))

    l = 0
    for point_a, point_b in zip(segment[:-1, :3], segment[1:, :3]):
        l += np.linalg.norm(point_b - point_a)

    return segment


def generate_vessel_tree():
    ica = generate_vessel_segment(
        length=30 + np.random.randn(),
        radius_intervals=[
            (80 + np.random.randint(-10, 10), 3),
            (160 + np.random.randint(-10, 10), 4),
            (200 + np.random.randint(-10, 10), 3.5),
            (256, 2.5),
        ],
        radius_init=1,
        tortuosity=0.4,
        initial_direction=(0.3, 0.3, 1),
        curve_bias=(0.5, 0.01, 0.001),
    )
    aca = generate_vessel_segment(
        length=20 + 0.6 * np.random.randn(),
        origin=ica[-1],
        radius_intervals=[(80 + np.random.randint(-10, 10), 2), (256, 1.5)],
        radius_init=2.1,
        tortuosity=0.5,
        initial_direction=(-1, 0.3, 0.4),
        curve_bias=(0.01, 0.01, 0.005),
    )
    m1 = generate_vessel_segment(
        length=20 + 2 * np.random.randn(),
        origin=ica[-1],
        radius_intervals=[(80 + np.random.randint(-10, 10), 2), (256, 1.5)],
        radius_init=2.2,
        tortuosity=0.3,
        initial_direction=(1, 0, 0.4),
        curve_bias=(0.005, 0.005, 0.5),
    )
    m2a = generate_vessel_segment(
        length=20 + 3 * np.random.randn(),
        origin=m1[-1],
        radius_intervals=[(256, 1)],
        radius_init=1.4,
        tortuosity=0.5,
        initial_direction=(1, 1, 0.2),
        curve_bias=(0.1, 0.2, 0.1),
    )
    m2b = generate_vessel_segment(
        length=20 + 3 * np.random.randn(),
        origin=m1[-1],
        radius_intervals=[(256, 1)],
        radius_init=1.4,
        tortuosity=0.3,
        initial_direction=(1, -1, 0.2),
        curve_bias=(0.4, 0.02, 0.2),
    )

    return np.stack((ica, aca, m1, m2a, m2b), axis=0)


def generate_dummy_dataset(num_samples: int = 200, path=""):
    dataset = np.stack([generate_vessel_tree() for _ in range(num_samples)], axis=0)

    path = os.path.join(path, "data.npy")

    np.save("data_train.npy", dataset)


def main():
    generate_dummy_dataset()
    # tree = generate_vessel_tree()

    # segment = np.concatenate(tree, axis=0)

    # segment[..., :3] -= segment[..., :3].mean(0, keepdims=True)
    # max_r = np.max(np.abs(segment))

    # points, radii = segment[..., :3], segment[..., 3]

    # fig = plt.figure(figsize=(10, 8))

    # ax = fig.add_subplot(projection="3d")
    # ax.scatter(0, 0, 0, color="red", s=50)
    # ax.scatter(*points[...,].T, s=(10 * radii) ** 2, c=points[..., 2])

    # ax.set_xlim(-max_r, max_r)
    # ax.set_ylim(-max_r, max_r)
    # ax.set_zlim(-max_r, max_r)

    # plt.show()


if __name__ == "__main__":
    main()
