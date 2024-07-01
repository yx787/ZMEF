# from __future__ import print_function

import time
import yaml
import warnings
import argparse

from tqdm import tqdm
from model import MEF
from util.logger import Logger
from util.common_utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from util.dataset import *
from util.mefssim import MEF_MSSSIM
from util.Loss import GradientLoss, mef_loss


torch.cuda.set_device(1)

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("cpu")


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/test.yaml')
    args = parser.parse_args()

    return args


def main(config, log, writer):

    ##################################################### dataset ######################################################

    if config['dataset'] == 'MEFB':
        test_set = MEFB(config['data_path'])
    elif config['dataset'] == 'MEF':
        test_set = MEF_dataset(config['data_path'])
    else:
        raise NotImplementedError

    loader = DataLoader(test_set, batch_size=1, shuffle=False)
    loss = MEF_MSSSIM(is_lum=False)
    grad = GradientLoss()

    for (img, imgs, img_name) in loader:

        log.logger.info(img_name)
        img = img.to(device)
        imgs = imgs.to(device)

        n, c, h, w = img.shape
        config['img_size'][0] = h
        config['img_size'][1] = w

        ################################################## network #####################################################
        netx = MEF().to(device)

        ############################################### optimizer ######################################################

        total_parameters = sum([param.nelement() for param in netx.parameters()])
        parameters = [{'params': netx.parameters()}]
        optimizer = torch.optim.Adam(parameters, lr=config['lr'])

        # FLOPs1, params1 = profile(model=net1, inputs=net_input1)
        # FLOPs2, params2 = profile(model=net2, inputs=net_input2)
        #
        log.logger.info("Total parameters: %.2fM" % (total_parameters / 1e6))
        # log.logger.info("FLOPs: %.2fG" % ((FLOPs1+FLOPs2) / 1e9))
        # log.logger.info("params: %.2fM" % ((params1+params2) / 1e6))

        # using multi-step as the learning rate change strategy
        # scheduler = MultiStepLR(optimizer, milestones=[350], gamma=0.1)  # learning rates

        ################################################ start iteration ###############################################

        for step in tqdm(range(config['num_iter'])):
            optimizer.zero_grad()

            # get the network output
            output = netx(img)

            grad_loss = grad.getloss(output, imgs)
            loss_ssim = 1 - mef_loss(output, imgs.squeeze(0))
            losses = loss_ssim + 0.1 * grad_loss
            losses.backward()
            optimizer.step()

            # write to tensorboard
            #writer.add_scalar(img_name + "/loss_mef_ssim", str(losses), step)
            # writer.add_scalar(img_name + "/psnr", losses['psnr'], step)
            # change the learning rate
            # scheduler.step(step)

            ############################################### saving #####################################################

            if (step + 1) % config['save_freq'] == 0:
                print("MEF_SSIM_C: {}; loss_ssim: {}; loss_grad: {}".format(1 - loss_ssim.data, loss_ssim.data, grad_loss))
                with torch.no_grad():
                    # remove the padding area
                    # out_x_np = out_x_np[:, padh//2:padh//2+img_size_input[1], padw//2:padw//2+img_size_input[2]]

                    save_path_x = os.path.join(config['save_path']+config['experiment_name'], '%s_%d_x.png' % (img_name[0], step + 1))
                    out_x_np = np.uint8(255 * np.transpose(torch_to_np(output).squeeze(), (1, 2, 0)))
                    cv2.imwrite(save_path_x, out_x_np)
        cv2.imwrite('./results_SICE/{}.png'.format(img_name[0]), out_x_np)

                    # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))


if __name__ == '__main__':

    ################################################### preperation ####################################################

    args = parse()

    with open(args.config, mode='r') as f:
        config = yaml.load(f)

    set_random_seed(config['rand_seed'])

    os.makedirs(config['save_path']+config['experiment_name'], exist_ok=True)
    os.makedirs(config['writer_path']+config['experiment_name'], exist_ok=True)
    writer = SummaryWriter(config['writer_path']+config['experiment_name'])
    log = Logger(filename=config['save_path']+config['experiment_name']+'.log', level='debug')

    config_str = ""
    for k, v in config.items():
        config_str += '\n\t{:<30}: {}'.format(k, str(v))
    log.logger.info(config_str)

    main(config, log, writer)

    writer.close()
    log.logger.info("all done at %s" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
