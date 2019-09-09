import numpy.testing as npt

from mmdet.utils.flops_counter import params_to_string


def test_params_to_string():
    npt.assert_equal(params_to_string(1e9), '1000.0 M')
    npt.assert_equal(params_to_string(2e5), '200.0 k')
    npt.assert_equal(params_to_string(3e-9), '3e-09')
