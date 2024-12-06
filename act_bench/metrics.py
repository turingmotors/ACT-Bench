import numpy as np


def final_displacement_error(
    estimated_trajectory: np.ndarray,
    template_trajectory: np.ndarray,
) -> float | list[float]:
    assert estimated_trajectory.ndim >= 2

    if estimated_trajectory.ndim == 2:
        fde = np.sqrt(((estimated_trajectory[-1] - template_trajectory[-1]) ** 2).sum())
        return fde

    fdes = []
    for i in range(len(estimated_trajectory)):
        fde = np.sqrt(((estimated_trajectory[i, -1] - template_trajectory[i, -1]) ** 2).sum())
        fdes.append(fde)
    return fdes


def average_displacement_error(
    estimated_trajectory: np.ndarray,
    template_trajectory: np.ndarray,
) -> float | list[float]:
    assert estimated_trajectory.ndim >= 2

    if estimated_trajectory.ndim == 2:
        ade = np.sqrt(((estimated_trajectory - template_trajectory) ** 2).sum(axis=1)).mean()
        return ade

    ades = []
    for i in range(len(estimated_trajectory)):
        ade = np.sqrt(((estimated_trajectory[i] - template_trajectory[i]) ** 2).sum(axis=1)).mean()
        ades.append(ade)
    return ades
