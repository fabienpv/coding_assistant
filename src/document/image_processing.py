from PIL import Image, PngImagePlugin
import numpy as np
import cv2 as cv
import hashlib
import glob
import os

from typing import Union


def PIL_to_opencv(im):
    """Convert a PIL Image to an OpenCV image.
    
        :param im: PIL Image.
        :return: OpenCV image.
        :rtype: numpy.ndarray"""
    return cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)


def opencv_to_PIL(array):
    """Convert a OpenCV array to a PIL Image.
    
        :param numpy.ndarray array: The input OpenCV array.
        :return: A PIL Image.
        :rtype: PIL.Image.Image"""
    return Image.fromarray(cv.cvtColor(array, cv.COLOR_BGR2RGB))


def horizontal_stretch(image: 'Image.Image', coef: float | int = 1.5) -> 'Image':
    """Stretch the image horizontally.
    
        :param Image.Image image: The input image.
        :param float coef: The stretch coefficient. Defaults to 1.5.
        :return: The horizontally stretched image.
        :rtype: Image.Image"""
    return image.resize((int(image.size[0] * coef), image.size[1]))


def auto_return(array: np.ndarray, return_array: bool, cv_convert: bool = False):
    """Return the input array or convert it to PIL Image.
    
        :param np.ndarray array: Input array.
        :param bool return_array: If True, return the array directly.
        :param bool cv_convert: If True, convert from OpenCV to PIL.
        :return: The array or PIL Image.
        :rtype: Union[np.ndarray, PIL.Image.Image]"""
    if return_array:
        return array
    else:
        if cv_convert:
            return opencv_to_PIL(array)
        else:
            return Image.fromarray(array)

def auto_converter_PIL_to_opencv(img: Union['Image.Image', 'np.ndarray']):
    """Convert image to OpenCV format if necessary.
    
        :param img: Image to convert. Can be a PIL Image or a NumPy ndarray.
        :type img: Union[Image.Image, np.ndarray]
        :return: Image in OpenCV format (NumPy ndarray).
        :rtype: np.ndarray
        :raises TypeError: If the image format is not supported."""
    if type(img) is PngImagePlugin.PngImageFile or type(img) is Image.Image:
        img = PIL_to_opencv(img)
    elif type(img) is np.ndarray:
        print("img is already a ndarray")
    else:
        raise TypeError(f"image format {type(img)} not supported in auto_converter_PIL_image")
    return img


def laplacian(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Apply the Laplacian filter to an image.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image.Image', 'np.ndarray']
        :param return_array: Whether to return the result as a NumPy array.
        :type return_array: bool, optional
        :return: The Laplacian filtered image.
        :rtype: Union['Image.Image', 'np.ndarray']"""
    img = auto_converter_PIL_to_opencv(img)
    img = cv.Laplacian(img, ddepth=-1)
    img = np.uint16(img)
    img = img ** 2 / 20
    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    return auto_return(array=img, return_array=return_array, cv_convert=True)

def laplacian_text_enhancer(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Enhance text in an image using Laplacian filtering.
    
        Args:
            img (Union[Image.Image, np.ndarray]): Input image.
            return_array (bool, optional): Whether to return a NumPy array.
                Defaults to False.
    
        Returns:
            Union[Image.Image, np.ndarray]: Enhanced image."""
    img = laplacian(img, return_array=True)
    img = 255 - img
    mask = (img[:, :, 0] < 245) | (img[:, :, 1] < 245) | (img[:, :, 2] < 245)
    img[mask] = 0
    # img = salt_and_pepper_denoiser(img, return_array=True)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def canny(img: Union['Image', 'np.ndarray'], return_array: bool = False):
    """Applies the Canny edge detection algorithm to an image.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image', 'np.ndarray']
        :param return_array: Whether to return the result as a NumPy array,
            default is False.
        :type return_array: bool
        :return: The Canny edge detected image.
        :rtype: Union['Image', 'np.ndarray']
        """
    # https://stackoverflow.com/questions/62589819/why-do-we-convert-laplacian-to-uint8-in-opencv
    # https://docs.opencv.org/4.1.0/d4/d86/group__imgproc__filter.html#filter_depths
    img = auto_converter_PIL_to_opencv(img)
    img = cv.Canny(img, 50, 150)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def gaussian_thresholding(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Applies Gaussian adaptive thresholding to an image.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image.Image', 'np.ndarray']
        :param return_array: Whether to return the result as a NumPy array.
        :type return_array: bool, optional
        :return: Thresholded image (PIL Image or NumPy array).
        :rtype: Union['Image.Image', 'np.ndarray']"""
    img = auto_converter_PIL_to_opencv(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def thresholding_B(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Thresholds the image to zero all pixels below 180.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image.Image', 'np.ndarray']
        :param return_array: Whether to return the result as a NumPy array,
            default is False.
        :type return_array: bool
        :return: Thresholded image.
        :rtype: Union['Image.Image', 'np.ndarray']"""
    img = auto_converter_PIL_to_opencv(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 180, 255, cv.THRESH_TOZERO)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def thresholding_A(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Thresholds the input image to zero where pixel values are below 80.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image.Image', 'np.ndarray']
        :param return_array: Whether to return the result as a NumPy array.
        :type return_array: bool, optional
        :return: Thresholded image.
        :rtype: Union['Image.Image', 'np.ndarray']"""
    img = auto_converter_PIL_to_opencv(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 80, 255, cv.THRESH_TOZERO)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def contrast_enhancer(img: Union['Image.Image', 'np.ndarray'], return_array: bool = False):
    """Enhance the contrast of an image.
    
        :param img: Input image (PIL Image or NumPy array).
        :type img: Union['Image.Image', 'np.ndarray']
        :param return_array: Whether to return the image as a NumPy array,
            defaults to False.
        :type return_array: bool, optional
        :return: Enhanced image.
        :rtype: Union['Image.Image', 'np.ndarray']"""
    img = auto_converter_PIL_to_opencv(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = np.uint16(img)
    mask = img > 127
    img[mask] = np.clip(img[mask] ** 1.03, 0, 255)
    img = 255 - img
    mask = img > 127
    img[mask] = np.clip(img[mask] ** 1.03, 0, 255)
    img = np.uint8(255 - img)
    return auto_return(array=img, return_array=return_array, cv_convert=True)


def hash_from_image(img: 'Image.Image') -> str:
    """Generate an MD5 hash from an image.
    
        :param img: Image object.
        :type img: Image.Image
        :return: MD5 hash as a hexadecimal string.
        :rtype: str"""
    img_bytes = bytes(img.tobytes())
    return hashlib.md5(img_bytes).hexdigest()


def crop_image_on_bboxes(
        img: 'Image.Image', 
        bboxes: list, 
        save_folder: str = None,
        mpp: bool = False
    ) -> list[Union['Image.Image', str]]:
    """Crop image based given list of bboxes such as to create
        sub images, either returned or save locally.
    
        :param img: the image to crop:
        :param bboxes: list of dictionaries with ordered values: left, top,
            right, bottom
        :param save_folder: where to save the cropped image. If None, returns a
            list of 'Image.Image', else, returns a list of image paths (str)
        :param mpp: whether the bboxes concern Milestone Payment Plan (special
            tag in name)
        :return: list of images or image paths"""
    list_images = []
    for bbox in bboxes:
        _img_ = img.copy()
        crop_tuple = tuple(bbox.values())
        _img_ = _img_.crop(crop_tuple)
        if save_folder:
            file_name = f"{hash_from_image(_img_)}.jpg"
            if not save_folder.endswith("/"):
                save_folder += "/"
            if mpp:
                file_name = f"mpp_{file_name}"
                # remove already saved mpp to limit duplicates
                for mpp_path in glob.glob(save_folder + "mpp_"):
                    os.remove(mpp_path)
            saved_path = save_folder + file_name
            _img_.save(saved_path)
            list_images.append(saved_path)
        else:
            list_images.append(_img_)
    return list_images

    

if __name__ == "__main__":
    image = Image.open('C:/Users/Fabien.Pavy/Downloads/test_image_6.png')
    image = horizontal_stretch(image)
    # image = salt_and_pepper_denoiser(image)
    # image = laplacian_text_enhancer(image)
    # image = gaussian_thresholding(image)
    # image = thresholding_A(image)
    # image = thresholding_B(image)
    # image = salt_and_pepper_denoiser(image)
    # image = contrast_enhancer(image)
    image.show()
