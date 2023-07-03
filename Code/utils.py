import os
import re
import time
from datetime import datetime
from math import pi
from socket import gethostname

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.ndimage import filters, interpolation, measurements
from tqdm import tqdm


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def crop_image(img, d=32):
    """
    Make dimensions divisible by d

    :param pil img:
    :param d:
    :return:
    """

    new_size = (img.size[0] - img.size[0] % d, img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.

    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.0


def prepare_image(file_name, imsize=-1):
    """
    loads makes it divisible
    :param file_name:
    :param imsize:
    :return: the numpy representation of the image
    """
    img_pil = crop_image(get_image(file_name, imsize)[0], d=32)
    return pil_to_np(img_pil)


def create_augmentations(np_image):
    """
    convention: original, left, upside-down, right, rot1, rot2, rot3
    :param np_image:
    :return:
    """
    aug = [
        np_image.copy(),
        np.rot90(np_image, 1, (1, 2)).copy(),
        np.rot90(np_image, 2, (1, 2)).copy(),
        np.rot90(np_image, 3, (1, 2)).copy(),
    ]
    flipped = np_image[:, ::-1, :].copy()
    aug += [
        flipped.copy(),
        np.rot90(flipped, 1, (1, 2)).copy(),
        np.rot90(flipped, 2, (1, 2)).copy(),
        np.rot90(flipped, 3, (1, 2)).copy(),
    ]
    return aug


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def save_image(name, image_np, output_path="output/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images by concatenating them.
    :param images_np:
    :param nrow:
    :return:
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_img_for_tensorboard(suptitle, images_np, interpolation="lanczos"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """

    assert len(images_np) == 2
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [
        x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0)
        for x in images_np
    ]

    grid = get_image_grid(images_np, 2)
    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    fig = plt.figure()
    plt.suptitle(suptitle)
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap="gray", interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    return fig


def plot_feature_map_for_tensorboard(suptitle, images_np, interpolation="lanczos"):
    """
    Draws images in a grid

    Args:
        images_np: feature maps,an np.array of size No_featuresxHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """

    img_list = []
    for i in images_np:
        img_list.append(np.expand_dims(i, axis=0))
    assert len(img_list) == 8
    n_channels = max(x.shape[0] for x in img_list)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    img_list = [
        x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0)
        for x in img_list
    ]
    grid = get_image_grid(img_list, 8)
    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    fig = plt.figure(figsize=(4, 1.0), dpi=200)
    plt.suptitle(suptitle)
    plt.axis("off")
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap="gray", interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    # fig = plt.figure(figsize=(4, 1.), dpi=200)
    # plt.suptitle(suptitle)
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                  nrows_ncols=(1, 8),  # creates 2x2 grid of axes
    #                  axes_pad=0.1,  # pad between axes in inch.
    #                  )
    # for ax, im in zip(grid, img_list):
    #     # Iterating over the grid returns the Axes.
    #     im = np.clip(im * 255, 0, 255).astype(np.uint8)
    #     ax.imshow(im[0], cmap='gray', interpolation=interpolation)
    #     ax.axis('off')
    return fig


def plot_image_grid(name, images_np, interpolation="lanczos", output_path="output/"):
    """
    Draws images in a grid

    Args:
        images_np: list of images, each image is np.array of size 3xHxW or 1xHxW
        nrow: how many images will be in one row
        interpolation: interpolation used in plt.imshow
    """
    assert len(images_np) == 2
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    images_np = [
        x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0)
        for x in images_np
    ]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap="gray", interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)

    plt.savefig(output_path + "{}.png".format(name))


def save_graph(name, graph_list, output_path="output/"):
    plt.clf()
    plt.plot(graph_list)
    plt.savefig(output_path + name + ".png")


def save_feature_maps(name, images_np, output_path="output/"):
    img_list = []
    for i in images_np:
        img_list.append(np.expand_dims(i, axis=0))
    assert len(img_list) == 8
    n_channels = max(x.shape[0] for x in img_list)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"

    img_list = [
        x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0)
        for x in img_list
    ]
    grid = get_image_grid(img_list, 8)
    p = np_to_pil(grid)
    p.save(output_path + "{}.jpg".format(name))


def imresize(
    im,
    scale_factor=None,
    output_shape=None,
    kernel=None,
    antialiasing=True,
    kernel_shift_flag=False,
):
    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
    scale_factor, output_shape = fix_scale_and_size(
        im.shape, output_shape, scale_factor
    )

    # For a given numeric kernel case, just do convolution and sub-sampling (downscaling only)
    if type(kernel) == np.ndarray and scale_factor[0] <= 1:
        return numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag)

    # Choose interpolation method, each method has the matching kernel size
    method, kernel_width = {
        "cubic": (cubic, 4.0),
        "lanczos2": (lanczos2, 4.0),
        "lanczos3": (lanczos3, 6.0),
        "box": (box, 1.0),
        "linear": (linear, 2.0),
        None: (cubic, 4.0),  # set default interpolation method as cubic
    }.get(kernel)

    # Antialiasing is only used when downscaling
    antialiasing *= scale_factor[0] < 1

    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
    sorted_dims = np.argsort(np.array(scale_factor)).tolist()

    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
    out_im = np.copy(im)
    for dim in sorted_dims:
        # No point doing calculations for scale-factor 1. nothing will happen anyway
        if scale_factor[dim] == 1.0:
            continue

        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
        # weights that multiply the values there to get its result.
        weights, field_of_view = contributions(
            im.shape[dim],
            output_shape[dim],
            scale_factor[dim],
            method,
            kernel_width,
            antialiasing,
        )

        # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
        out_im = resize_along_dim(out_im, dim, weights, field_of_view)

    return out_im


def fix_scale_and_size(input_shape, output_shape, scale_factor):
    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
    # same size as the number of input dimensions)
    if scale_factor is not None:
        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
        if np.isscalar(scale_factor):
            scale_factor = [scale_factor, scale_factor]

        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
    # to all the unspecified dimensions
    if output_shape is not None:
        output_shape = list(np.uint(np.array(output_shape))) + list(
            input_shape[len(output_shape) :]
        )

    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
    # sub-optimal, because there can be different scales to the same output-shape.
    if scale_factor is None:
        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

    # Dealing with missing output-shape. calculating according to scale-factor
    if output_shape is None:
        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

    return scale_factor, output_shape


def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
    # such that each position from the field_of_view will be multiplied with a matching filter from the
    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
    # around it. This is only done for one dimension of the image.

    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
    # 1/sf. this means filtering is more 'low-pass filter'.
    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
    kernel_width *= 1.0 / scale if antialiasing else 1.0

    # These are the coordinates of the output image
    out_coordinates = np.arange(1, out_length + 1)

    # These are the matching positions of the output-coordinates on the input image coordinates.
    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
    match_coordinates = 1.0 * out_coordinates / scale + 0.5 * (1 - 1.0 / scale)

    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
    left_boundary = np.floor(match_coordinates - kernel_width / 2)

    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
    expanded_kernel_width = np.ceil(kernel_width) + 2

    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
    # vertical dim is the pixels it 'sees' (kernel_size + 2)
    field_of_view = np.squeeze(
        np.uint(
            np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1
        )
    )

    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
    # 'field_of_view')
    weights = fixed_kernel(
        1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1
    )

    # Normalize weights to sum up to 1. be careful from dividing by 0
    sum_weights = np.sum(weights, axis=1)
    sum_weights[sum_weights == 0] = 1.0
    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

    # We use this mirror structure as a trick for reflection padding at the boundaries
    mirror = np.uint(
        np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1)))
    )
    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

    # Get rid of  weights and pixel positions that are of zero weight
    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
    weights = np.squeeze(weights[:, non_zero_out_pixels])
    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
    return weights, field_of_view


def resize_along_dim(im, dim, weights, field_of_view):
    # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
    tmp_im = np.swapaxes(im, dim, 0)

    # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
    # tmp_im[field_of_view.T], (bsxfun style)
    weights = np.reshape(weights.T, list(weights.T.shape) + (np.ndim(im) - 1) * [1])

    # This is a bit of a complicated multiplication: tmp_im[field_of_view.T] is a tensor of order image_dims+1.
    # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
    # only, this is why it only adds 1 dim to the shape). We then multiply, for each pixel, its set of positions with
    # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
    # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
    # same number
    tmp_out_im = np.sum(tmp_im[field_of_view.T] * weights, axis=0)

    # Finally we swap back the axes to the original order
    return np.swapaxes(tmp_out_im, dim, 0)


def numeric_kernel(im, kernel, scale_factor, output_shape, kernel_shift_flag):
    # See kernel_shift function to understand what this is
    if kernel_shift_flag:
        kernel = kernel_shift(kernel, scale_factor)

    # First run a correlation (convolution with flipped kernel)
    out_im = np.zeros_like(im)
    for channel in range(np.ndim(im)):
        out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)

    # Then subsample and return
    return out_im[
        np.round(
            np.linspace(0, im.shape[0] - 1 / scale_factor[0], output_shape[0])
        ).astype(int)[:, None],
        np.round(
            np.linspace(0, im.shape[1] - 1 / scale_factor[1], output_shape[1])
        ).astype(int),
        :,
    ]


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between od and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (
        sf - (kernel.shape[0] % 2)
    )

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(shift_vec))) + 1, "constant")

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


# These next functions are all interpolation methods. x is the distance from the left pixel center


def cubic(x):
    absx = np.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
    ) * ((1 < absx) & (absx <= 2))


def lanczos2(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps)
    ) * (abs(x) < 2)


def box(x):
    return ((-0.5 <= x) & (x < 0.5)) * 1.0


def lanczos3(x):
    return (
        (np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps)
        / ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps)
    ) * (abs(x) < 3)


def linear(x):
    return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))


def np_imresize(
    im,
    scale_factor=None,
    output_shape=None,
    kernel=None,
    antialiasing=True,
    kernel_shift_flag=False,
):
    return np.clip(
        imresize(
            im.transpose(1, 2, 0),
            scale_factor,
            output_shape,
            kernel,
            antialiasing,
            kernel_shift_flag,
        ).transpose(2, 0, 1),
        0,
        1,
    )


##############################################################################################################
# define project log path creation function
##############################################################################################################
# create project log path
def create_project_log_path(project_path, **kwargs):
    # year_month_day/hour_min/(model_log_dir, model_checkpoint_dir, tensorboard_log_dir)/
    date = datetime.now()
    program_time = project_path + date.strftime("%Y%m%d-%H%M%S")

    Readme_flag = False
    if "Readme" in kwargs.keys():
        Readme_flag = True
    if Readme_flag:
        readme = kwargs.pop("Readme")

    program_log_parent_dir = program_time
    for key, value in kwargs.items():
        program_log_parent_dir = (
            program_log_parent_dir + "_" + key + "_{}".format(value)
        )

    program_log_parent_dir = program_log_parent_dir + "/"
    if not os.path.exists(program_log_parent_dir):
        os.mkdir(program_log_parent_dir)

    # model checkpoint dir
    model_checkpoint_dir = program_log_parent_dir + "model_checkpoint_dir/"
    if not os.path.exists(model_checkpoint_dir):
        os.mkdir(model_checkpoint_dir)

    # tensorboard_log_dir
    tensorboard_log_dir = program_log_parent_dir + "tensorboard_log_dir/"
    if not os.path.exists(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)

    # model_log_dir
    model_log_dir = program_log_parent_dir + "model_log_dir/"
    if not os.path.exists(model_log_dir):
        os.mkdir(model_log_dir)

    # write exp log
    if Readme_flag:
        with open(program_log_parent_dir + "Readme.txt", "w") as f:
            f.write(readme + "\r\n")
            for key, value in kwargs.items():
                f.write(key + ": {}\r\n".format(value))
            f.write("program log dir: " + program_log_parent_dir + "\r\n")

    return (
        program_log_parent_dir,
        model_checkpoint_dir,
        tensorboard_log_dir,
        model_log_dir,
    )


# write summary to readme.txt
def summary2readme(summary, readme_path):
    with open(readme_path, "a") as fh:
        fh.write(summary + "\r\n")