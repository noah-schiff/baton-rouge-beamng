import pytest

from brmap.geo import Extent


@pytest.fixture
def extent():
    """The full Baton Rouge box from the spec (16,384 m square)."""
    return Extent(-91.160, 30.430, 8192.0)


@pytest.fixture
def small_extent():
    """A 1,024 m box around the same center, for fast raster tests."""
    return Extent(-91.160, 30.430, 512.0)
