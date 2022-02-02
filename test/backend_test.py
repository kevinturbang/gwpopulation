import unittest

import numpy
import pytest

import gwpopulation


class TestSetBackend(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_unsupported_backend_raises_value_error(self):
        with self.assertRaises(ValueError):
            gwpopulation.set_backend("fail")

    def test_set_backend_numpy(self):
        gwpopulation.set_backend("numpy")
        from gwpopulation.cupy_utils import xp

        self.assertEqual(xp, numpy)

    def test_set_backend_jax(self):
        pytest.importorskip("jax.numpy")
        import jax.numpy as jnp

        gwpopulation.set_backend("jax.numpy")
        from gwpopulation.cupy_utils import xp

        self.assertEqual(jnp, xp)

    def test_enable_cupy_deprecated(self):
        with pytest.deprecated_call():
            gwpopulation.enable_cupy()

    def test_disable_cupy_deprecated(self):
        with pytest.deprecated_call():
            gwpopulation.disable_cupy()
