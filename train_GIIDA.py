import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from network import ImageClassifier
from loss import consistency_loss, get_trans_matrix_prob, reg
from data.prepare_data_da import generate_dataloader as Dataloader
from visual import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def opts():

    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='UDA')
    ## dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) + ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--transform_type', type=str,
                        default='randomcrop', help='randomcrop | randomsizedcrop | center')
    parser.add_argument('--strongaug', action='store_true', default=True,
                        help='whether use the strong augmentation (i.e., RandomAug) it is True in FixMatch and UDA')
    parser.add_argument('--mu', type=int, default=2, help='unlabeled batch size / labeled batch size')

    ## model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=architecture_names,
                        help='backbone architecture: ' + ' | '.join(architecture_names) + ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--temperature', default=1.8, type=float, help='parameter temperature scaling')
    parser.add_argument('--trade-off1', default=0.5, type=float,
                        help='fixmatch loss')
    parser.add_argument('--trade-off2', default=1.0, type=float,
                        help='hyper-parameter for correct loss')
    parser.add_argument('--trade-off3', default=0.5, type=float,
                        help='hyper-parameter for regularization')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--early', default=40, type=int, metavar='N', help='number of total epochs to early stopping')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument('--q-length', type=int, default=3, help="queue length")

    ## log parameters
    parser.add_argument("--log", type=str, default='GIIDA',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    ## visual parameters
    parser.add_argument('--visual-T', action='store_true', help="visualization transmatrix")
    parser.add_argument('--img-path', type=str, default='img', help="save path of transmatrix")


    args = parser.parse_args()

    data_dir = '/data/' ### change your own data root
    args.root = data_dir + args.root

    if args.data == 'OfficeHome':
        args.num_class = 65
    elif args.data == 'Office31':
        args.num_class = 31
    elif args.data == 'DomainNet':
        args.num_class = 345
    elif args.data == 'VisDA2017':
        args.num_class = 12
        args.category_mean = True
        args.transform_type = 'center'
        print('training with VisDA2017')
    return args

def main():

    args = opts()
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    ###### data
    dataloaders = Dataloader(args)
    train_source_loader = dataloaders['source']
    train_target_loader = dataloaders['target']
    val_loader = dataloaders['trans_test']
    test_loader = val_loader
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    ##### create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, args.num_class, bottleneck_dim=args.bottleneck_dim).to(device)

    ##### optimizer and lr scheduler
    optimizer = torch.optim.SGD(classifier.get_parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    #### init matrix & Category queues
    global transMatrix, tgt_queue_img, tgt_queue_prob
    transMatrix = (torch.eye(args.num_class)).to(device)
    tgt_queue_img = [[] for i in range(args.num_class)]
    tgt_queue_prob = [[] for i in range(args.num_class)]

    #### resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    #### analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return
    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    #### start training
    best_acc1 = 0.
    for epoch in range(min(args.epochs, args.early)):
        print("lr:", lr_scheduler.get_last_lr()[0])
        train(train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, args)
        acc1 = validate(val_loader, classifier, args)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()

def update_queue(x_t, max_prob, labels_t, t_queue_img, t_queue_prob, q_length):
    for c in range(len(t_queue_img)):
        ind_t = torch.where(labels_t == c)[0]
        t_queue_img[c].extend(x_t[ind_t])
        t_queue_prob[c].extend(max_prob[ind_t])
        while len(t_queue_img[c]) > q_length:
            t_queue_img[c].pop(0)
            t_queue_prob[c].pop(0)

def mixup_align(x_s, labels_s, t_queue_img, t_queue_prob, ratio):
    x_mix_list, labels_mix_list = [], []
    for s in range(x_s.size(0)):
        if len(tgt_queue_img[labels_s[s].item()]) != 0:
            weight_norm = F.normalize(torch.tensor(t_queue_prob[labels_s[s].item()]).to(device), p=1, dim=-1)
            ind_t = torch.multinomial(weight_norm, 1)  # Probabilistic sampling
            x_t_choice = t_queue_img[labels_s[s].item()][ind_t]  # selected tgt sample
            x_mix_align = (1 - ratio) * x_s[s] + ratio * x_t_choice

            x_mix_list.append(x_mix_align)
            labels_mix_list.append(labels_s[s])

    x_mix = torch.stack(x_mix_list, 0)
    labels_mix = torch.stack(labels_mix_list, 0)
    return x_mix, labels_mix

def train(train_source_iter, train_target_iter, model, optimizer, lr_scheduler, epoch, args):

    global transMatrix, tgt_queue_img, tgt_queue_prob

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    reg_losses = AverageMeter('Reg Loss', ':3.2f')
    y_m_losses_correct = AverageMeter('correct_m loss', ':3.2f')
    ssl_losses = AverageMeter('SSL Ls', ':3.2f')
    cls_accs = AverageMeter('s_Acc', ':3.1f')
    tgt_accs = AverageMeter('t_Acc', ':3.1f')

    progress = ProgressMeter(args.iters_per_epoch,
        [losses, y_m_losses_correct, ssl_losses, cls_accs, tgt_accs, reg_losses],
        prefix="Epoch: [{}]".format(epoch))

    # ratio
    ratio = (epoch + 1) * 0.1 if epoch < 10 else 1.0
    print("ratio: " + str(ratio))

    # switch to train mode
    model.train()
    end = time.time()
    for i in range(args.iters_per_epoch):

        (x_s,_), labels_s = next(train_source_iter)
        (x_t, x_t_u), labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_u, f_t_u = model(x_t_u)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t, dim=1), dim=-1)
        ssl_loss, _, = consistency_loss(y_t, y_t_u, T=1.0, p_cutoff=0.97)  ## fixmatch loss

        # construct intermediate domain samples
        update_queue(x_t, max_prob, pred_u, tgt_queue_img, tgt_queue_prob, args.q_length)
        x_mix, labels_mix = mixup_align(x_s, labels_s, tgt_queue_img, tgt_queue_prob, ratio)
        y_mix, f_mix = model(x_mix)

        ### train noise classifier: closed form
        f_n = f_mix
        pseudo_label_n = labels_mix
        f_n_norm = f_n / (torch.norm(f_n, dim=-1).reshape(f_n.shape[0], 1))
        f_n_kernel = torch.clamp(f_n_norm.mm(f_n_norm.transpose(dim0=1, dim1=0)), -0.99999999, 0.99999999)
        soft_label_n = 0.999 * torch.nn.functional.one_hot(pseudo_label_n, args.num_class) + 0.001 / float(args.num_class)

        ### source noise class posterior estimate
        f_s_norm = f_s / (torch.norm(f_s, dim=-1).reshape(f_s.shape[0], 1))
        f_s_kernel = torch.clamp(f_s_norm.mm(f_n_norm.transpose(dim0=1, dim1=0)), -0.99999999, 0.99999999)
        class_poster_s = f_s_kernel.mm(
            torch.inverse(f_n_kernel + 0.001 * torch.eye(f_n_kernel.size(0)).to(device))).mm(soft_label_n)
        class_poster_s = torch.clamp(class_poster_s, 0.00000001, 0.99999999)
        class_poster_s = class_poster_s / (torch.sum(class_poster_s, dim=1).reshape(class_poster_s.shape[0], 1))

        ### intermediate domain noise class posterior estimate
        f_m_norm = f_mix / (torch.norm(f_mix, dim=-1).reshape(f_mix.shape[0], 1))
        f_m_kernel = torch.clamp(f_m_norm.mm(f_n_norm.transpose(dim0=1, dim1=0)), -0.99999999, 0.99999999)
        class_poster_m = f_m_kernel.mm(
            torch.inverse(f_n_kernel + 0.001 * torch.eye(f_n_kernel.size(0)).to(device))).mm(soft_label_n)
        class_poster_m = torch.clamp(class_poster_m, 0.00000001, 0.99999999)
        class_poster_m = class_poster_m / (torch.sum(class_poster_m, dim=1).reshape(class_poster_m.shape[0], 1))

        ### update T
        tM_current = get_trans_matrix_prob(class_poster_s, labels_s, args.num_class)
        ind = torch.where(torch.sum(tM_current, dim=-1) != 0)[0]
        transMatrix[ind] = 0.99 * transMatrix[ind] + 0.01 * tM_current[ind].detach()

        ### clean to noise
        y_m_softmax = F.softmax(y_mix, dim=1)
        y_m_correct = y_m_softmax.mm(transMatrix.detach())
        y_m_loss_correct = nn.KLDivLoss()(torch.log(y_m_correct), class_poster_m.detach())  ## correct loss

        # classification loss & regularization loss
        cls_loss = F.cross_entropy(y_s, labels_s)
        reg_loss = reg(y_t, args.temperature)/(args.batch_size * args.mu)

        ### compute total loss
        loss = cls_loss + args.trade_off1 * ssl_loss + \
               args.trade_off2 * y_m_loss_correct + \
               args.trade_off3 * reg_loss


        ####print
        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))
        reg_losses.update(reg_loss.item(), x_s.size(0))
        y_m_losses_correct.update(y_m_loss_correct.item(), x_s.size(0))
        ssl_losses.update(ssl_loss.item(), x_s.size(0))

        ## backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    if args.visual_T:
        plot_Matrix(epoch, transMatrix.detach().cpu(), args.num_class, path=args.img_path)

def validate(val_loader, model, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        # classes = val_loader.dataset.classes
        classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife',
               'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))
    return top1.avg


if __name__ == '__main__':
    main()