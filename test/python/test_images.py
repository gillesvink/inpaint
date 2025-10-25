from pathlib import Path
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import inpaint
import numpy as np


def compare_image_arrays(a: PngImageFile, b: PngImageFile, tolerance: float):
    """Simple comparison util for the images"""
    if a.width != b.width or a.height != b.height:
        raise ValueError("Images differ in dimensions.")

    a_array = np.asarray(a, dtype=np.float32) / 255.0
    b_array = np.asarray(b, dtype=np.float32) / 255.0

    close_elements = np.isclose(a_array, b_array, atol=tolerance)

    similar_count = np.sum(close_elements)
    total_elements = a_array.size

    percentage = (similar_count / total_elements) * 100

    return percentage


@pytest.mark.parametrize(
    ("input_image", "input_mask", "expected_image"),
    [
        (
            "./test/images/input/bird.png",
            "./test/images/mask/thin.png",
            "./test/images/expected/telea/bird_thin.png",
        ),
        (
            "./test/images/input/bird.png",
            "./test/images/mask/medium.png",
            "./test/images/expected/telea/bird_medium.png",
        ),
        (
            "./test/images/input/bird.png",
            "./test/images/mask/large.png",
            "./test/images/expected/telea/bird_large.png",
        ),
        (
            "./test/images/input/bird.png",
            "./test/images/mask/text.png",
            "./test/images/expected/telea/bird_text.png",
        ),
        (
            "./test/images/input/toad.png",
            "./test/images/mask/thin.png",
            "./test/images/expected/telea/toad_thin.png",
        ),
        (
            "./test/images/input/toad.png",
            "./test/images/mask/medium.png",
            "./test/images/expected/telea/toad_medium.png",
        ),
        (
            "./test/images/input/toad.png",
            "./test/images/mask/text.png",
            "./test/images/expected/telea/toad_text.png",
        ),
    ],
)
def test_images(input_image: Path, input_mask: Path, expected_image: Path) -> None:
    """Test images as also tested in Rust package, to confirm the bridge is working as expected"""
    image = Image.open(input_image)
    mask = Image.open(input_mask)
    expected = Image.open(expected_image)

    output = inpaint.telea(image, mask)

    result = compare_image_arrays(output, expected, 0.005)
    assert result == 100.0
