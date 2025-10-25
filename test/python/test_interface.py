import pytest
from PIL import Image
import inpaint
import numpy as np


@pytest.mark.parametrize(("input_type"), [np.float32, np.uint32, np.int32])
def test_numpy_interface(input_type) -> None:
    """Make sure return type is correct, as telea inpaint supports both images and arrays

    Args:
        input_type:: type for array, should match return value
    """
    test_image = Image.open("./test/images/input/bird.png")
    test_array = np.asarray(test_image, dtype=np.float32) / 256.0
    test_array = np.asarray(test_image, dtype=input_type)

    output = inpaint.telea(test_array, test_array[:, :, 0])

    assert type(output) is type(test_array)
