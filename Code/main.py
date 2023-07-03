import itertools
import queue
from collections import namedtuple
from threading import Thread
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from Denoisers import non_local_means
from net.losses import ExclusionLoss
from net.noise import get_noise
from net.skip_model import multiple_output_skip
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from utils import (create_augmentations, create_project_log_path, np_to_torch,
                   plot_feature_map_for_tensorboard, plot_img_for_tensorboard,
                   save_feature_maps, save_graph, save_image, summary2readme,
                   torch_to_np, prepare_image, plot_image_grid, auto_select_GPU)


class BITSNET(object):
    def __init__(
        self,
        image,
        clean_img1,
        clean_img2,
        model_to_run,
        output_excl_loss_weight,
        use_feature_loss,
        feature_excl_loss_weight,
        lambda_r,
        lambda_t,
        learning_rate,
        project_path,
        device,
        plot_during_training=True,
        show_every=500,
        num_iter=8000,
        original_reflection=None,
        original_transmission=None,
    ):
        self.image = image
        self.clean_img1 = clean_img1
        self.clean_img2 = clean_img2
        self.model_to_run = model_to_run  # 1: 'DoubleDIP', 2: 'BITSNET'
        self.use_feature_loss = use_feature_loss  # True or False
        self.feature_excl_loss_weight = feature_excl_loss_weight
        self.output_excl_loss_weight = output_excl_loss_weight  # default value 0.01
        self.project_path = project_path
        self.device = device
        self.plot_during_training = plot_during_training

        self.psnrs = []
        self.show_every = show_every
        self.image_name = "mixed_sample"
        self.num_iter = num_iter  # 6000 from DeepRED, 8000 from Double DIP
        self.loss_function = None
        self.parameters = None
        self.learning_rate = learning_rate  # 0.008 from DeepRED, 0.0005 from Double DIP
        self.input_depth = clean_img1.shape[0]
        self.reflection_net_inputs = None
        self.transmission_net_inputs = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.mix_recon_loss = None
        self.total_loss = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None

        # RED params
        # denoiser_function     - an external denoiser function, used as black box, this function
        # sigma_f_r/t           - the sigma to send the denoiser function for reflection or transmission
        # update_iter           - denoised image updated every 'update_iter' iteration
        # method              - 'fixed_point' or 'grad' or 'mixed'
        #
        # # equation params #
        #   lamb_r, lamb_t      - regularization parameter for RED, (beta in the original code)
        #   mu_r, mu_t          - ADMM parameter
        #   LR_R, LR_T          - learning rate of the parameter x, needed only if method!=fixed point
        #
        if self.model_to_run == 2:
            self.denoiser_function = non_local_means
            self.sigma_f_r = 3
            self.sigma_f_t = 3
            self.update_iter = 10
            self.method = "fixed_point"  # method: 'fixed_point' or 'grad' or 'mixed'
            self.lamb_r = lambda_r  # default 0.5
            self.lamb_t = lambda_t  # default 0.5
            self.mu_r = 0.5
            self.mu_t = 0.5
            self.LR_R = None
            self.LR_T = None

        # must get numpy noisy image, and return numpy denoised image
        self._init_all()

    def _init_all(self):
        self._init_exp_log()
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        self.images = create_augmentations(self.image)
        self.images_torch = [
            np_to_torch(image).to(self.device) for image in self.images
        ]

        save_image("img1.png", self.clean_img1, output_path=self.model_log_dir)
        save_image("img2.png", self.clean_img2, output_path=self.model_log_dir)
        save_image("mix.png", self.image, output_path=self.model_log_dir)

    def _init_inputs(self):
        input_type = "noise"
        origin_noise = torch_to_np(
            get_noise(
                self.input_depth,
                input_type,
                (self.images_torch[0].shape[2], self.images_torch[0].shape[3]),
            )
            .to(self.device)
            .detach()
        )
        np.save(self.model_checkpoint_dir + "reflection_noise_input.npy", origin_noise)
        self.reflection_net_inputs = [
            np_to_torch(aug).to(self.device).detach()
            for aug in create_augmentations(origin_noise)
        ]

        origin_noise = torch_to_np(
            get_noise(
                self.input_depth,
                input_type,
                (self.images_torch[0].shape[2], self.images_torch[0].shape[3]),
            )
            .to(self.device)
            .detach()
        )
        np.save(self.model_checkpoint_dir + "transmission_net_input.npy", origin_noise)
        self.transmission_net_inputs = [
            np_to_torch(aug).to(self.device).detach()
            for aug in create_augmentations(origin_noise)
        ]

        if self.model_to_run == 2:  # 1: 'DoubleDIP', 2: 'BITSNET'

            def __denoise_job(q, f, sigma):
                while True:
                    paras = q.get()
                    img_idx = paras[0]  # The image index in the augmented image list.
                    img = paras[1]  # The image (in numpy)
                    output_image_list = paras[
                        2
                    ]  # The list to take output of denoising function
                    temp_denoised = f(img, sigma)
                    output_image_list[img_idx] = temp_denoised
                    q.task_done()

            # init denoiser multi-thread
            self.reflection_Rs = [
                np.zeros_like(r) for r in create_augmentations(origin_noise)
            ]
            self.u_Rs = [np.zeros_like(r) for r in create_augmentations(origin_noise)]
            self.f_Rs = self.reflection_Rs.copy()
            self.R_queue = queue.Queue()  # The queue save a list of denoised images.
            for i in range(8):  # Create 8 threads
                R_denoiser_thread = Thread(
                    target=__denoise_job,
                    args=(
                        self.R_queue,
                        self.denoiser_function,
                        self.sigma_f_r,
                    ),
                    daemon=True,
                )
                R_denoiser_thread.start()
            # construct the input list to the queue for thread.
            thread_input_list = [
                [idx, i_r_Rs.copy(), self.f_Rs]
                for idx, i_r_Rs in enumerate(self.reflection_Rs)
            ]
            for i_thread_input in thread_input_list:
                self.R_queue.put(i_thread_input)

            self.transmission_Ts = [
                np.zeros_like(r) for r in create_augmentations(origin_noise)
            ]
            self.u_Ts = [np.zeros_like(r) for r in create_augmentations(origin_noise)]
            self.f_Ts = self.transmission_Ts.copy()
            self.T_queue = queue.Queue()
            for i in range(8):  # Create 8 threads
                T_denoiser_thread = Thread(
                    target=__denoise_job,
                    args=(
                        self.T_queue,
                        self.denoiser_function,
                        self.sigma_f_t,
                    ),
                    daemon=True,
                )
                T_denoiser_thread.start()
            # construct the input list to the queue for thread.
            thread_input_list = [
                [idx, i_t_Ts.copy(), self.f_Ts]
                for idx, i_t_Ts in enumerate(self.transmission_Ts)
            ]
            for i_thread_input in thread_input_list:
                self.T_queue.put(i_thread_input)

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + [
            p for p in self.transmission_net.parameters()
        ]

    def _init_nets(self):
        pad = "reflection"

        reflection_net = multiple_output_skip(
            self.input_depth,
            self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode="bilinear",
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        )

        transmission_net = multiple_output_skip(
            self.input_depth,
            self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode="bilinear",
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            act_fun="LeakyReLU",
        )

        self.reflection_net = reflection_net.to(self.device)
        self.transmission_net = transmission_net.to(self.device)

    def _init_losses(self):
        self.l1_loss = nn.L1Loss().to(self.device)
        self.exclusion_loss = ExclusionLoss().to(self.device)
        if self.model_to_run == 2:
            self.l2_loss = nn.MSELoss().to(self.device)

    def _init_exp_log(self):
        if self.model_to_run == 1:
            model = "DoubleDIP"
        elif self.model_to_run == 2:
            model = "BITSNET"
        exp_params = {
            "model": model,
            "output_excl_loss_weight": self.output_excl_loss_weight,
            "use feature loss": self.use_feature_loss,
            "optimiser": "ADAM",
            "num_iter": self.num_iter,
            "learning_rate": self.learning_rate,
        }
        if self.use_feature_loss:
            exp_params["feature_excl_loss_weights"] = self.feature_excl_loss_weight

        if self.model_to_run == 2:
            exp_params["lamb_r"] = self.lamb_r
            exp_params["lamb_t"] = self.lamb_t
        Readme = (
            "In this experiment, we will try to run the "
            + model
            + " .\r\n"
            + "exp_params:\r\n"
            + str(exp_params)
            + "\r\n"
        )

        kwargs = {
            "Readme": Readme,
            "model": model,
            "use_feature_loss": self.use_feature_loss,
            "lr": self.learning_rate,
        }
        if self.model_to_run == 2:
            kwargs["lamb"] = self.lamb_r
        if self.use_feature_loss:
            kwargs["FExcl_Lossweight"] = self.feature_excl_loss_weight

        (
            self.program_log_parent_dir,
            self.model_checkpoint_dir,
            self.tensorboard_log_dir,
            self.model_log_dir,
        ) = create_project_log_path(project_path=self.project_path, **kwargs)

    def optimize(self):
        if self.model_to_run == 1:
            self.train_doubleDIP()
        elif self.model_to_run == 2:
            self.train_via_admm()

    def train_doubleDIP(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        # define tensorboard writer
        self.writer = SummaryWriter(self.tensorboard_log_dir)
        with trange(self.num_iter, unit="epoch") as tepoch:
            for j in tepoch:
                optimizer.zero_grad()
                self._optimization_closure(j)
                self._obtain_current_result(j)
                self._write_to_tensorboard(j)
                if self.plot_during_training:
                    self._plot_closure(j, tepoch)
                optimizer.step()

        # close the writer after use
        self.writer.flush()
        self.writer.close()

    def train_via_admm(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # x update method:
        if self.method == "fixed_point":
            swap_iter = self.num_iter
            self.LR_R = None
            self.LR_T = None
            self.LR_MIX = None
        elif self.method == "grad":
            swap_iter = -1
        elif self.method == "mixed":
            swap_iter = self.num_iter // 2
        else:
            assert False, "method can only be 'fixed_point' or 'grad' or 'mixed' !"

        # get optimizer and loss function:
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        # define tensorboard writer
        self.writer = SummaryWriter(self.tensorboard_log_dir)

        # start ADMM iterations
        with trange(self.num_iter, unit="epoch") as tepoch:
            for i in tepoch:
                # step 1, update network
                optimizer.zero_grad()
                self._optimization_closure(i)
                optimizer.step()

                # step 2, update x using a denoiser and result from step 1
                self._obtain_current_result(i)
                if i % self.update_iter == 0:
                    self.R_queue.join()
                    self.T_queue.join()
                    thread_input_list = [
                        [idx, i_r_Rs.copy(), self.f_Rs]
                        for idx, i_r_Rs in enumerate(self.reflection_Rs)
                    ]
                    for i_thread_input in thread_input_list:
                        self.R_queue.put(i_thread_input)
                    thread_input_list = [
                        [idx, i_t_Ts.copy(), self.f_Ts]
                        for idx, i_t_Ts in enumerate(self.transmission_Ts)
                    ]
                    for i_thread_input in thread_input_list:
                        self.T_queue.put(i_thread_input)
                    self.R_queue.join()
                    self.T_queue.join()

                # step 2, update x using a the denoiser (f_x) and network outputs (out_np)
                if i < swap_iter:
                    self.reflection_Rs[self.aug] = (
                        1
                        / (self.lamb_r + self.mu_r)
                        * (
                            self.lamb_r * self.f_Rs[self.aug]
                            + self.mu_r
                            * (self.current_result.reflection + self.u_Rs[self.aug])
                        )
                    )  # eq. 11 in the article
                    self.transmission_Ts[self.aug] = (
                        1
                        / (self.lamb_t + self.mu_t)
                        * (
                            self.lamb_t * self.f_Ts[self.aug]
                            + self.mu_t
                            * (self.current_result.transmission + self.u_Ts[self.aug])
                        )
                    )  # eq. 11 in the article

                else:
                    self.reflection_Rs[self.aug] = self.reflection_Rs[
                        self.aug
                    ] - self.LR_R * (
                        self.lamb_r
                        * (self.reflection_Rs[self.aug] - self.f_Rs[self.aug])
                        + self.mu_r
                        * (
                            self.reflection_Rs[self.aug]
                            - self.current_result.reflection
                            - self.u_Rs[self.aug]
                        )
                    )  # eq. 12 in the article
                    self.transmission_Ts[self.aug] = self.transmission_Ts[
                        self.aug
                    ] - self.LR_T * (
                        self.lamb_t
                        * (self.transmission_Ts[self.aug] - self.f_Ts[self.aug])
                        + self.mu_t
                        * (
                            self.transmission_Ts[self.aug]
                            - self.current_result.transmission
                            - self.u_Ts[self.aug]
                        )
                    )  # eq. 12 in the article

                np.clip(
                    self.reflection_Rs[self.aug], 0, 1, out=self.reflection_Rs[self.aug]
                )  # making sure that image is in bounds
                np.clip(
                    self.transmission_Ts[self.aug],
                    0,
                    1,
                    out=self.transmission_Ts[self.aug],
                )  # making sure that image is in bounds

                # step 3, update u
                self.u_Rs[self.aug] = (
                    self.u_Rs[self.aug]
                    + self.current_result.reflection
                    - self.reflection_Rs[self.aug]
                )
                self.u_Ts[self.aug] = (
                    self.u_Ts[self.aug]
                    + self.current_result.transmission
                    - self.transmission_Ts[self.aug]
                )

                # write to tensorboard and print results
                self._write_to_tensorboard(i)
                if self.plot_during_training:
                    self._plot_closure(i, tepoch)

        # close the writer after use
        self.writer.flush()
        self.writer.close()

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, step):
        if step == self.num_iter - 1:
            reg_noise_std = 0
        elif step < 1000:
            reg_noise_std = (1 / 1000.0) * (step // 100)
        else:
            reg_noise_std = 1 / 1000.0
        self.aug = self._get_augmentation(step)
        if step == self.num_iter - 1:
            self.aug = 0
        reflection_net_input = self.reflection_net_inputs[self.aug] + (
            self.reflection_net_inputs[self.aug].clone().normal_() * reg_noise_std
        )
        transmission_net_input = self.transmission_net_inputs[self.aug] + (
            self.transmission_net_inputs[self.aug].clone().normal_() * reg_noise_std
        )

        self.reflection_feature, self.reflection_out = self.reflection_net(
            reflection_net_input
        )
        self.transmission_feature, self.transmission_out = self.transmission_net(
            transmission_net_input
        )

        self.mix_recon_loss = self.l1_loss(
            self.reflection_out + self.transmission_out, self.images_torch[self.aug]
        )
        self.total_loss = (
            self.mix_recon_loss
            + self.output_excl_loss_weight
            * self.exclusion_loss(self.reflection_out, self.transmission_out)
        )

        if self.model_to_run == 2:
            # add the term from ADMM
            self.total_loss += self.mu_r * self.l2_loss(
                self.reflection_out,
                np_to_torch(self.reflection_Rs[self.aug] - self.u_Rs[self.aug]).to(
                    self.device
                ),
            )
            self.total_loss += self.mu_t * self.l2_loss(
                self.transmission_out,
                np_to_torch(self.transmission_Ts[self.aug] - self.u_Ts[self.aug]).to(
                    self.device
                ),
            )
        if self.use_feature_loss:
            # add the term from feature exclusion loss
            self.total_loss += self.feature_excl_loss_weight * self.exclusion_loss(
                torch.sigmoid(
                    self.reflection_feature[
                        :, torch.randperm(self.reflection_feature.size()[1]), :, :
                    ]
                ),
                torch.sigmoid(
                    self.transmission_feature[
                        :, torch.randperm(self.transmission_feature.size()[1]), :, :
                    ]
                ),
            )

        self.total_loss.backward()

    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        # if step == self.num_iter - 1 or step % 8 == 0:
        # obtain the output in np
        reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
        transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
        psnr = peak_signal_noise_ratio(
            self.images[0], reflection_out_np + transmission_out_np
        )
        self.psnrs.append(psnr)

        # obtain the feature in np
        transmission_feature_np = np.clip(torch_to_np(self.transmission_feature), 0, 1)
        reflection_feature_np = np.clip(torch_to_np(self.reflection_feature), 0, 1)

        # puts in self.current result the current result.
        self.current_result = SeparationResult(
            reflection=reflection_out_np,
            transmission=transmission_out_np,
            psnr=psnr,
            transmission_feature=transmission_feature_np,
            reflection_feature=reflection_feature_np,
        )

        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
            torch.save(
                self.transmission_net,
                self.model_checkpoint_dir + "transmission_net.pth",
            )
            torch.save(
                self.reflection_net, self.model_checkpoint_dir + "reflection_net.pth"
            )

    def _write_to_tensorboard(self, step):
        # write the metrics into tensorboard
        if step == self.num_iter - 1 or step % (16 * 25) == 0:
            # ------------------if have ground truth-------------------------------------------------------------------
            # calc the metrics for clean_img1
            if mean_squared_error(
                self.clean_img1, self.current_result.reflection * 2
            ) < mean_squared_error(
                self.clean_img1, self.current_result.transmission * 2
            ):
                est_img1 = self.current_result.reflection * 2
                est_img1_feature_map = self.current_result.reflection_feature
            else:
                est_img1 = self.current_result.transmission * 2
                est_img1_feature_map = self.current_result.transmission_feature

            psnr_img1 = peak_signal_noise_ratio(self.clean_img1, est_img1)
            ssim_img1 = ssim(
                self.clean_img1.transpose(1, 2, 0),
                est_img1.transpose(1, 2, 0),
                channel_axis=2,
                data_range=est_img1.max() - est_img1.min()
            )

            # calc the metrics for clean_img2
            if mean_squared_error(
                self.clean_img2, self.current_result.reflection * 2
            ) < mean_squared_error(
                self.clean_img2, self.current_result.transmission * 2
            ):
                est_img2 = self.current_result.reflection * 2
                est_img2_feature_map = self.current_result.reflection_feature
            else:
                est_img2 = self.current_result.transmission * 2
                est_img2_feature_map = self.current_result.transmission_feature

            psnr_img2 = peak_signal_noise_ratio(self.clean_img2, est_img2)
            ssim_img2 = ssim(
                self.clean_img2.transpose(1, 2, 0),
                est_img2.transpose(1, 2, 0),
                channel_axis=2,
                data_range=est_img2.max() - est_img2.min()
            )

            # ---------------------if not have ground truth--------------------------------------------------------------
            # est_img1 = self.current_result.reflection * 2
            # est_img1_feature_map = self.current_result.reflection_feature
            # est_img2 = self.current_result.transmission * 2
            # est_img2_feature_map = self.current_result.transmission_feature
            # psnr_img1 = peak_signal_noise_ratio(self.clean_img1, est_img1)
            # ssim_img1 = ssim(self.clean_img1.transpose(1, 2, 0), est_img1.transpose(1, 2, 0), multichannel=True)
            # psnr_img2 = peak_signal_noise_ratio(self.clean_img2, est_img2)
            # ssim_img2 = ssim(self.clean_img2.transpose(1, 2, 0), est_img2.transpose(1, 2, 0), multichannel=True)
            # -----------------------------------------------------------------------------------------------------------

            # write the metrics into tensorboard
            self.writer.add_scalar("total loss", self.total_loss.item(), step)
            self.writer.add_scalar("mix recon loss", self.mix_recon_loss.item(), step)
            self.writer.add_scalar("PSNR of Mix", self.current_result.psnr, step)
            self.writer.add_scalar("PSNR of img1", psnr_img1, step)
            self.writer.add_scalar("PSNR of img2", psnr_img2, step)
            self.writer.add_scalar("SSIM of img1", ssim_img1, step)
            self.writer.add_scalar("SSIM of img2", ssim_img2, step)

            # write summary into readme
            summary = (
                "Iteration {:5d}    total Loss: {:5f} Mix Recon Loss: {:5f}  "
                "PSNR_Mix: {:f}  PSNR_img1: {:f}  PSNR_img2: {:f}"
                "  SSIM_img1: {:f}  SSIM_img2: {:f}".format(
                    step,
                    self.total_loss.item(),
                    self.mix_recon_loss.item(),
                    self.current_result.psnr,
                    psnr_img1,
                    psnr_img2,
                    ssim_img1,
                    ssim_img2,
                )
            )
            summary2readme(summary, self.program_log_parent_dir + "Readme.txt")

            # write the imgs into tensorboard
            if step == self.num_iter - 1 or step % (16 * 25 * 2) == 0:
                # write the output img of DIP
                fig = plot_img_for_tensorboard(
                    "step %d: mix PSNR %f" % (step, self.current_result.psnr),
                    [
                        self.images[0],
                        self.current_result.reflection
                        + self.current_result.transmission,
                    ],
                )
                self.writer.add_figure("mix", fig, step, close=True)
                plt.close("all")
                fig = plot_img_for_tensorboard(
                    "step %d: img1 PSNR %f" % (step, psnr_img1),
                    [self.clean_img1, est_img1],
                )
                self.writer.add_figure("img1", fig, step, close=True)
                plt.close("all")
                fig = plot_img_for_tensorboard(
                    "step %d: img2 PSNR %f" % (step, psnr_img2),
                    [self.clean_img2, est_img2],
                )
                self.writer.add_figure("img2", fig, step, close=True)
                plt.close("all")

                # write the feature map of DIP
                fig = plot_feature_map_for_tensorboard(
                    "img1 feature map", est_img1_feature_map
                )
                self.writer.add_figure("img1 feature map", fig, step, close=True)
                plt.close("all")
                fig = plot_feature_map_for_tensorboard(
                    "img2 feature map", est_img2_feature_map
                )
                self.writer.add_figure("img2 feature map", fig, step, close=True)
                plt.close("all")

    def _plot_closure(self, step, tepoch):
        tepoch.set_description(
            f"model {self.model_to_run} feature loss {self.use_feature_loss} Epoch {step}"
        )
        tepoch.set_postfix(
            loss=self.total_loss.item(), PSNR_gt=self.current_result.psnr
        )
        if step % self.show_every == self.show_every - 1:
            plot_image_grid(
                name="left_right_{}".format(step),
                images_np=[
                    self.current_result.reflection,
                    self.current_result.transmission,
                ],
                output_path=self.tensorboard_log_dir,
            )

    def finalize(self):
        save_graph(
            self.image_name + "_psnr", self.psnrs, output_path=self.tensorboard_log_dir
        )
        save_image(
            self.image_name + "_reflection.png",
            self.best_result.reflection,
            output_path=self.tensorboard_log_dir,
        )
        save_image(
            self.image_name + "_transmission.png",
            self.best_result.transmission,
            output_path=self.tensorboard_log_dir,
        )
        save_image(
            self.image_name + "_reflection2.png",
            2 * self.best_result.reflection,
            output_path=self.tensorboard_log_dir,
        )
        save_image(
            self.image_name + "_transmission2.png",
            2 * self.best_result.transmission,
            output_path=self.tensorboard_log_dir,
        )
        save_image(
            self.image_name + "_original.png",
            self.images[0],
            output_path=self.tensorboard_log_dir,
        )

        # save feature map
        save_feature_maps(
            self.image_name + "_transmission_feature",
            self.best_result.transmission_feature,
            output_path=self.tensorboard_log_dir,
        )
        save_feature_maps(
            self.image_name + "reflection_feature",
            self.best_result.reflection_feature,
            output_path=self.tensorboard_log_dir,
        )


SeparationResult = namedtuple(
    "SeparationResult",
    [
        "reflection",
        "transmission",
        "psnr",
        "reflection_feature",
        "transmission_feature",
    ],
)


def main():
    # choose sepcifc img and separte it

    # define the thread func
    def process_func(
        mix,
        t1,
        t2,
        model_to_run,
        output_excl_loss_weight,
        use_feature_loss,
        feature_excl_loss_weight,
        lambda_r,
        lambda_t,
        learning_rate,
        project_path,
    ):
        # choose a GPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # run the algorithm
        s = BITSNET(
            mix,
            t1,
            t2,
            model_to_run,
            output_excl_loss_weight,
            use_feature_loss,
            feature_excl_loss_weight,
            lambda_r,
            lambda_t,
            learning_rate,
            project_path,
            device,
        )
        s.optimize()
        s.finalize()
        del s

    thread_nums = 0
    #  run in multiple process
    processes = []

    img1_dir = "../images/set4_00.jpg"
    img2_dir = "../images/set4_01.jpg"

    t1 = prepare_image(img1_dir, 400)
    t2 = prepare_image(img2_dir, 400)
    mix = (t1 + t2) / 2

    # define the hyper-params
    model_to_runs = [2]  # 1: 'DoubleDIP', 2: 'BITSNET'
    output_excl_loss_weights = [0.01]  # 0.01 from Double-DIP
    use_feature_losses = [True]
    feature_excl_loss_weights = [1e-3]  # [1e-2, 1e-3, 1e-4]
    lambdas = [0.5]  # weight of RED, default 0.5
    learning_rates = [0.008]  # 0.008 from DeepRED, 0.0005 from Double DIP

    for (
        run,
        model_to_run,
        output_excl_loss_weight,
        use_feature_loss,
        feature_excl_loss_weight,
        lambda_red,
        learning_rate,
    ) in itertools.product(
        range(5),
        model_to_runs,
        output_excl_loss_weights,
        use_feature_losses,
        feature_excl_loss_weights,
        lambdas,
        learning_rates,
    ):
        project_path = "../result/"

        p = mp.Process(
            target=process_func,
            args=(
                mix,
                t1,
                t2,
                model_to_run,
                output_excl_loss_weight,
                use_feature_loss,
                feature_excl_loss_weight,
                lambda_red,
                lambda_red,
                learning_rate,
                project_path,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(10)
        thread_nums += 1
    if thread_nums >= 4:
        # close process
        for p in processes:
            p.join()
        processes = []
        thread_nums = 0

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
