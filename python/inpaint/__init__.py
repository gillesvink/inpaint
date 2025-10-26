from PIL.Image import Image
import PIL
from .inpaint import (
    telea_inpaint,
)
import numpy as np
from typing import Union


class InpaintError(Exception):
    """Raised when inpainting failed for whatever reason."""


def telea(
    image: Union[Image, np.array],
    mask: Union[Image, np.array],
    radius: int = 5,
) -> Union[Image, np.array]:
    """Inpaint the image using the Telea algorithm

    Args:
        image: image to inpaint
        mask: image containing black and white mask for region to inpaint
        radius (optional): radius of near pixels that are considered for
                           inpainting. Defaults to 5.

    Raises:
        InpaintError: if something goes unrecoverably wrong.

    Returns:
        inpainted image
    """

    is_pil_image = isinstance(image, PIL.Image.Image)

    image_array = np.array(image) if is_pil_image else image
    mask_array = np.array(mask)[:, :, 0] if is_pil_image else mask

    original_image_type = image_array.dtype
    image_array = _convert_to_float(image_array)
    mask_array = _convert_to_float(mask_array)

    try:
        output: np.array = telea_inpaint(
            image_array,
            mask_array,
            radius,
        )
    except RuntimeError as error: 
        raise InpaintError(str(error)) from error

    if not np.issubdtype(original_image_type, np.floating):
        output *= np.iinfo(original_image_type).max

    if is_pil_image:
        return PIL.Image.fromarray(output.astype(original_image_type))

    return output


def _convert_to_float(image_array: np.array):
    """Convert

    Args:
        image_array: array to convert to float
    """

    if not np.issubdtype(image_array.dtype, np.floating):
        image_array = image_array.astype(np.float32, copy=False) / float(
            np.iinfo(
                image_array.dtype,
            ).max
        )

    return image_array
