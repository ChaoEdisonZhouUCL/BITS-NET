import numpy as np
from bm3d import bm3d
from skimage.restoration import denoise_nl_means


def non_local_means(noisy_np_img, sigma, fast_mode=True):
    """ get a numpy noisy image
        returns a denoised numpy image using Non-Local-Means
    """
    sigma = sigma / 255.
    h = 0.6 * sigma if fast_mode else 0.8 * sigma
    patch_kw = dict(h=h,  # Cut-off distance, a higher h results in a smoother image
                    sigma=sigma,  # sigma provided
                    fast_mode=fast_mode,  # If True, a fast version is used. If False, the original version is used.
                    patch_size=5,  # 5x5 patches (Size of patches used for denoising.)
                    patch_distance=6,  # 13x13 search area
                    # multichannel=False
                    )
    if isinstance(noisy_np_img, list):
        denoised_imgs = []
        for i in noisy_np_img:
            denoised_img = []
            n_channels = i.shape[0]
            for c in range(n_channels):
                denoise_fast = denoise_nl_means(i[c, :, :], **patch_kw)
                denoised_img += [denoise_fast]
            denoised_imgs.append(np.array(denoised_img, dtype=np.float32))
        return denoised_imgs
    else:
        denoised_img = []
        n_channels = noisy_np_img.shape[0]
        for c in range(n_channels):
            denoise_fast = denoise_nl_means(noisy_np_img[c, :, :], **patch_kw)
            denoised_img += [denoise_fast]
        return np.array(denoised_img, dtype=np.float32)


def bm3d_denoiser(noisy_np_img, sigma):
    sigma = sigma / 255.
    return bm3d(noisy_np_img, sigma)
