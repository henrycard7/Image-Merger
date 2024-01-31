import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
        :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
        :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    #get image dimenstions
    rows, cols = image.shape[:2]
    # flip kernel for proper convolution values
    kernel = np.flip(kernel)
    kx = kernel.shape[0]
    ky = kernel.shape[1]
    #find padding
    xpad = kx // 2
    ypad = ky // 2
    #create output kernel
    output = np.zeros(image.shape)

    # if grey-scale
    if len(image.shape) == 2:
        paddedImage = np.pad(image, ((xpad, xpad), (ypad, ypad)))
        # loop through rows and columns of image
        for i in range(0, cols):
            for j in range(0, rows):
                # vector multiplication of image section and kernel
                output[j, i] = (paddedImage[j:j + kx, i:i + ky] * kernel).sum()

    # if rgb
    else:
        paddedImage = np.pad(image, ((xpad, xpad), (ypad, ypad), (0, 0)))
        # loop through rgb values
        for c in range(3):
            # loop through rows and columns of image
            for i in range(0, cols):
                for j in range(0, rows):
                    # vector multiplication of image section and kernel
                    output[j, i, c] = (paddedImage[j:j + kx, i:i + ky, c] * kernel).sum()

    return output
