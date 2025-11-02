from ctg.utils import compute_time_slabs


def test_compute_time_slabs():
    slabs = compute_time_slabs(0.0, 0.1, 0.03)
    assert slabs[0] == (0.0, 0.03)
    assert slabs[-1][1] > 0.09
