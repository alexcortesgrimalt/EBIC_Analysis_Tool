import numpy as np
from code.perpendicular import calculate_perpendicular_profiles


def test_calculate_perpendicular_profiles_basic():
    # create a simple horizontal line from (2,5) to (8,5)
    line = ((2.0, 5.0), (8.0, 5.0))
    sem = np.zeros((10, 12), dtype=float)
    cur = np.zeros_like(sem)
    # put a bright spot at center of each perp
    sem[5, 5] = 1.0

    profiles = calculate_perpendicular_profiles(line, num_lines=3, length_um=1.0,
                                                 sem_data=sem, current_data=cur,
                                                 pixel_size_m=1e-6)
    assert isinstance(profiles, list)
    assert len(profiles) == 3
    for p in profiles:
        assert 'sem' in p and 'current' in p and 'dist_um' in p
        assert len(p['sem']) == len(p['current'])
