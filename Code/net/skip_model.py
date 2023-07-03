import torch
import torch.nn as nn

from net.layers import *


def skip(
        num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3,
        filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
    """
    Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        # model_tmp.add(layer_norm(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            # model_tmp.add(layer_norm(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model


class multiple_output_skip(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3,
                 filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
        super(multiple_output_skip, self).__init__()
        outlayer = '8'
        net = skip(num_input_channels, num_output_channels, num_channels_down,
                   num_channels_up,
                   num_channels_skip, filter_size_down,
                   filter_size_up, filter_skip_size, need_sigmoid, need_bias,
                   pad, upsample_mode, downsample_mode, act_fun, need1x1_up)
        self.feature_list = []
        self.output_list = []
        flag = True
        for n, c in net.named_children():
            if flag:
                self.feature_list.append(c)
            else:
                self.output_list.append(c)
            if n == outlayer:
                flag = False
        self.feature_net = nn.Sequential(*self.feature_list)
        self.output_net = nn.Sequential(*self.output_list)
        net = None

    def forward(self, input):
        feature = self.feature_net(input)
        output = self.output_net(feature)
        return feature, output


class multiple_output_multiple_features_skip(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128],
                 num_channels_up=[16, 32, 64, 128, 128],
                 num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3,
                 filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
        super(multiple_output_multiple_features_skip, self).__init__()
        outlayers = ['5', '8']
        base_net = skip(num_input_channels, num_output_channels, num_channels_down,
                        num_channels_up,
                        num_channels_skip, filter_size_down,
                        filter_size_up, filter_skip_size, need_sigmoid, need_bias,
                        pad, upsample_mode, downsample_mode, act_fun, need1x1_up)
        temp_net_list = []
        feature_net_list = []
        for n, c in base_net.named_children():
            temp_net_list.append(c)
            if n in outlayers:
                feature_net_list.append(temp_net_list)
                temp_net_list = []

        feature_net_list.append(temp_net_list)

        self.net_list = []
        for net in feature_net_list:
            self.net_list.append(nn.Sequential(*net))
        self.No_nets = len(self.net_list)
        self.net_list = nn.ModuleList(self.net_list)
        base_net = None

    def forward(self, input):
        features = []
        features.append(self.net_list[0](input))
        for i in range(1, self.No_nets):
            temp = features[-1]
            features.append(self.net_list[i](temp))

        return features[:-1], features[-1]


def skip_mask(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    """
    Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad,
                        downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        # model_tmp.add(layer_norm(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    return model


if __name__ == '__main__':
    from Code.Modules.util_torch import auto_select_GPU
    from Code.DoubleDIP.net.noise import get_noise

    device, _ = auto_select_GPU()

    # define input
    input_type = 'noise'
    origin_noise = get_noise(3, input_type, (400, 400)).to(device).detach()

    # define net
    pad = 'reflection'

    reflection_net = multiple_output_multiple_features_skip(
        3, 3,
        num_channels_down=[8, 16, 32, 64, 128],
        num_channels_up=[8, 16, 32, 64, 128],
        num_channels_skip=[0, 0, 0, 4, 4],
        upsample_mode='bilinear',
        filter_size_down=5,
        filter_size_up=5,
        need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').to(device)

    # run the net
    features, output = reflection_net(origin_noise)
    features = torch.from_numpy(np.array(features, dtype=np.float))
    print(len(features))
    print(features[0].shape)
    print('Done')
