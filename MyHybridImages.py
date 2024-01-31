import math
import numpy as np

from MyConvolution import convolve

def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
        :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour
    shape=(rows,cols,channels))
        :type numpy.ndarray
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage :type float
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage
    before subtraction to create the high-pass filtered image
    :type float
    :returns returns the hybrid image created
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
    a high-pass image created by subtracting highImage from highImage convolved with
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
        :rtype numpy.ndarray
    """
    # get kernels for each image
    lowPassKernel = makeGaussianKernel(lowSigma)
    highPassKernel = makeGaussianKernel(highSigma)
    #normalise each image and use convolution
    lowfImage = convolve(lowImage/255, lowPassKernel)
    highfImage = (highImage / 255) - convolve(highImage/255, highPassKernel)
    #combine images and rescale
    hybridImage = (lowfImage + highfImage) * 255
    return hybridImage



def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
     Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0, and the size should be floor(8*sigma+1) or
    floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    #given sigma function
    size = int(8 * sigma +1)
    if size % 2 == 0:
        size= size + 1

    #create grid
    xs = np.linspace(-(size//2),size//2, size)
    ys = np.linspace(-(size // 2), size // 2, size)
    x,y = np.meshgrid(xs,ys)

    #calculates Gaussian function values for grid
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    #normalises the kernel: ensures it sums to 1
    kernel = kernel / np.sum(kernel)
    return kernel


