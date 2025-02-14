# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import os
import math
import argparse
import numpy as np
from copy import deepcopy

from lib.env.quantize_env import QuantizeEnv
from lib.env.linear_quantize_env import LinearQuantizeEnv
from lib.rl.ddpg import DDPG

from lib.compress.CompressEval import CompressEval
#from lib.compress import Pruner
from lib.utils.data_utils import get_split_train_dataset, dataloader_to_tensor
from lib.compress.TrainingUtil import *

from tensorboardX import SummaryWriter

import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models
import models as customized_models

#import detectors
# import timm
from model import prc_model
from model import Denoising
from lib.compress.robustbench import load_cifar10, clean_accuracy, load_cifar10c
from autoattack import AutoAttack
import time
from lib.utils.utils import AverageMeter, accuracy
from progress.bar import Bar

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
print('support models: ', model_names)


def train(num_episode, agent, env, output, linear_quantization=False, debug=False):
    # best record
    best_reward = -math.inf
    best_policy = []

    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    while episode < num_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation), deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(num_episode / 10) == 0:
            agent.save_model(output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode

            if linear_quantization:
                if debug:
                    print('#{}: episode_reward:{:.4f} acc: {:.4f}, cost: {:.4f}'.format(episode, episode_reward,
                                                                                             info['accuracy'],
                                                                                             info['cost'] * 1. / 8e6))
                text_writer.write(
                    '#{}: episode_reward:{:.4f} acc: {:.4f}, cost: {:.4f}\n'.format(episode, episode_reward,
                                                                                         info['accuracy'],
                                                                                         info['cost'] * 1. / 8e6))
            else:
                if debug:
                    print('#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB'.format(episode, episode_reward,
                                                                                             info['accuracy'],
                                                                                             info['w_ratio']))
                text_writer.write(
                    '#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} MB\n'.format(episode, episode_reward,
                                                                                         info['accuracy'],
                                                                                         info['w_ratio']))

            final_reward = T[-1][0]
            # agent observe and update policy
            for i, (r_t, s_t, s_t1, a_t, done) in enumerate(T):
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for i in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy

            value_loss = agent.get_value_loss()
            policy_loss = agent.get_policy_loss()
            delta = agent.get_delta()
            tfwriter.add_scalar('reward/last', final_reward, episode)
            tfwriter.add_scalar('reward/best', best_reward, episode)
            tfwriter.add_scalar('info/accuracy', info['accuracy'], episode)
            tfwriter.add_text('info/best_policy', str(best_policy), episode)
            tfwriter.add_text('info/current_policy', str(env.strategy), episode)
            tfwriter.add_scalar('value_loss', value_loss, episode)
            tfwriter.add_scalar('policy_loss', policy_loss, episode)
            tfwriter.add_scalar('delta', delta, episode)
            if linear_quantization:
                tfwriter.add_scalar('info/coat_ratio', info['cost_ratio'], episode)
                # record the preserve rate for each layer
                for i, preserve_rate in enumerate(env.strategy):
                    tfwriter.add_scalar('preserve_rate_w/{}'.format(i), preserve_rate[0], episode)
                    tfwriter.add_scalar('preserve_rate_a/{}'.format(i), preserve_rate[1], episode)
            else:
                tfwriter.add_scalar('info/w_ratio', info['w_ratio'], episode)
                # record the preserve rate for each layer
                for i, preserve_rate in enumerate(env.strategy):
                    tfwriter.add_scalar('preserve_rate_w/{}'.format(i), preserve_rate, episode)

            text_writer.write('best reward: {}\n'.format(best_reward))
            text_writer.write('best policy: {}\n'.format(best_policy))
    text_writer.close()
    return best_policy, best_reward


def _validate(val_loader, model, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        criterion = nn.CrossEntropyLoss().cuda()

        t1 = time.time()
        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            bar = Bar('_validate:', max=len(val_loader))
            for i, (inputs, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                input_var, target_var = inputs.cuda(), targets.cuda()
                # noise = torch.randn_like(inputs, device='cuda') * 0.25
                # input_var += noise

                # compute output
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return top1.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Reinforcement Learning')

    parser.add_argument('--suffix', default=None, type=str, help='suffix to help you remember what experiment you ran')
    # env
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset to use')
    parser.add_argument('--dataset_root', default='data/imagenet', type=str, help='path to dataset')
    parser.add_argument('--preserve_ratio', default=0.1, type=float, help='preserve ratio of the model size')
    parser.add_argument('--min_bit', default=1, type=float, help='minimum bit to use')
    parser.add_argument('--max_bit', default=8, type=float, help='maximum bit to use')
    parser.add_argument('--float_bit', default=32, type=int, help='the bit of full precision float')
    parser.add_argument('--linear_quantization', dest='linear_quantization', action='store_true')
    parser.add_argument('--is_pruned', dest='is_pruned', action='store_true')
    # ddpg
    parser.add_argument('--hidden1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--lr_c', default=1e-3, type=float, help='learning rate for actor')
    parser.add_argument('--lr_a', default=1e-4, type=float, help='learning rate for actor')
    parser.add_argument('--warmup', default=20, type=int,
                        help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=1., type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=128, type=int, help='memory size for each layer')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.01, type=float, help='moving average for target network')
    # noise (truncated normal distribution)
    parser.add_argument('--init_delta', default=0.5, type=float,
                        help='initial variance of truncated normal distribution')
    parser.add_argument('--delta_decay', default=0.99, type=float,
                        help='delta decay during exploration')
    parser.add_argument('--n_update', default=1, type=int, help='number of rl to update each time')
    # training
    parser.add_argument('--max_episode_length', default=1e9, type=int, help='')
    parser.add_argument('--output', default='../../save', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--train_episode', default=600, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=234, type=int, help='')
    parser.add_argument('--n_worker', default=32, type=int, help='number of data loader worker')
    parser.add_argument('--data_bsize', default=256, type=int, help='number of data batch size')
    parser.add_argument('--finetune_epoch', default=1, type=int, help='')
    parser.add_argument('--finetune_gamma', default=0.8, type=float, help='finetune gamma')
    parser.add_argument('--finetune_lr', default=0.001, type=float, help='finetune gamma')
    parser.add_argument('--finetune_flag', default=True, type=bool, help='whether to finetune')
    parser.add_argument('--use_top5', default=False, type=bool, help='whether to use top5 acc in reward')
    parser.add_argument('--train_size', default=20000, type=int, help='number of train data size')
    parser.add_argument('--val_size', default=10000, type=int, help='number of val data size')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenet_v2', choices=model_names,
                    help='model architecture:' + ' | '.join(model_names) + ' (default: mobilenet_v2)')
    # device options
    parser.add_argument('--gpu_id', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    
    # compress options
    parser.add_argument('--quant_method', default='', type=str,
                        help='compress methods: ptq, ptqRFT, ptqNFT')
    
    parser.add_argument('--quant_level', default='', type=str,
                    help='quant_levels: int16,int8,int4,int2,int1')
    
    parser.add_argument('--prune_method', default='', type=str,
                    help='prune_methods: prune')
    
    parser.add_argument('--prune_level', default=1, type=float,
                help='prune_levels: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    
    args = parser.parse_args()
    base_folder_name = '{}_{}'.format(args.arch, args.dataset)
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    args.output = os.path.join(args.output, base_folder_name)
    tfwriter = SummaryWriter(logdir=args.output)
    text_writer = open(os.path.join(args.output, 'log.txt'), 'w')
    print('==> Output path: {}...'.format(args.output))

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'imagenet100':
        num_classes = 100
    elif args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'NWPU-RESISC45':
        num_classes = 45
        print("num_class=",num_classes)
    else:
        raise NotImplementedError

    device = torch.device("cuda")
    model = prc_model(num_classes=num_classes).cuda()
    #resnet18
    checkpoint = torch.load('/home/liuyang/workspace/HAQ/haq/Sehwag2021Proxy_R18.pt',map_location= device)
    # resnet50
    # checkpoint = torch.load('/home/liuyang/workspace/HAQ/haq/engstrom2019robustness.pt',map_location= device)
    new_checkpoint = {'backbone.' + k: v for k, v in checkpoint.items()}

    if args.dataset == 'cifar10':
        model.load_state_dict(new_checkpoint, strict=True)
    elif args.dataset == 'NWPU-RESISC45':
        checkpoint = torch.load('/home/liuyang/workspace/HAQ/haq/resnet19_NWPU_adv.pt',map_location= device)
        newcheckpoint = checkpoint['state_dict']
        model.load_state_dict(newcheckpoint, strict=True)
        # model_state_dict = model.state_dict()
        # for key, value in checkpoint.items():
        #     # 如果键在模型状态字典中，并且形状匹配，则添加到新字典中
        #     if key in model_state_dict and model_state_dict[key].shape == value.shape:
        #         new_checkpoint[key] = value
        # new_checkpoint = {'backbone.' + k: v for k, v in new_checkpoint.items()}
        # model.load_state_dict(new_checkpoint, strict=False)


    #加载数据集
    train_loader, val_loader, n_class = get_split_train_dataset(args.dataset, args.data_bsize, args.n_worker,
            val_size=args.val_size, train_size=args.train_size)
    

    _validate(val_loader=val_loader, model=model)

    #压缩前准确率
    # x_test, y_test = load_cifar10()
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    # acc = clean_accuracy(model, x_test, y_test)
    # print("clean_accuracy before compress",acc)

    # x_test, y_test = load_cifar10c(n_examples=1000, corruptions=['fog'], severity=5)
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    # acc = clean_accuracy(model, x_test, y_test)
    # print(f'CIFAR-10-C accuracy: {acc:.1%} before compress')

    #autoattack
    # x_test, y_test = load_cifar10(n_examples=50)
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    x_test, y_test = dataloader_to_tensor(val_loader,50)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    auto_robust_acc = adversary.run_standard_evaluation(x_test, y_test, robust_acc=True)
    print(f'Auto attack accuracy: {auto_robust_acc:.1%} before compress')


    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        #model = torch.nn.DataParallel(model).cuda()
        model.cuda()
        
    pretrained_model = deepcopy(model.state_dict())
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    cudnn.benchmark = True


    #模型压缩
    compressor = CompressEval(device,train_loader,val_loader,None,args.quant_level, args.quant_method,args.prune_level,args.prune_method,"results_file_name",model_directory="compress_model")
    compressed_model = compressor.single_prune_compress(model, model_name="prune_model", pruning_level=args.prune_level, pruning_methods=args.prune_method)
    pretrained_compress_model = deepcopy(compressed_model.state_dict())

    
    #压缩后准确率
    print("prune_level=", args.prune_level)
    _validate(val_loader=val_loader, model=compressed_model)

    # x_test, y_test = load_cifar10()
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    # acc = clean_accuracy(compressed_model, x_test, y_test)
    # print("clean_accuracy after compress",acc)

    # x_test, y_test = load_cifar10c(n_examples=1000, corruptions=['fog'], severity=5)
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    # acc = clean_accuracy(compressed_model, x_test, y_test)
    # print(f'CIFAR-10-C accuracy: {acc:.1%} after compress')

    #autoattack
    # x_test, y_test = load_cifar10(n_examples=50)
    # x_test = x_test.to(device)
    # y_test = y_test.to(device)
    x_test, y_test = dataloader_to_tensor(val_loader,50)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    adversary = AutoAttack(compressed_model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    auto_robust_acc = adversary.run_standard_evaluation(x_test, y_test, robust_acc=True)
    print(f'Auto attack accuracy: {auto_robust_acc:.1%} after compress')



    if args.linear_quantization:
        env = LinearQuantizeEnv(model, pretrained_model, args.dataset, args.dataset_root,
                                compress_ratio=args.preserve_ratio, n_data_worker=args.n_worker,
                                batch_size=args.data_bsize, args=args, float_bit=args.float_bit,
                                is_model_pruned=args.is_pruned)
    else:
        model_arch = prc_model(num_classes=45)
        env = QuantizeEnv(compressed_model, pretrained_compress_model, model_arch, args.dataset, args.dataset_root,
                          compress_ratio=args.preserve_ratio, n_data_worker=args.n_worker,
                          batch_size=args.data_bsize, args=args, float_bit=args.float_bit,
                          is_model_pruned=args.is_pruned)

    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1  # actions for weight and activation quantization
    args.rmsize = args.rmsize * len(env.quantizable_idx)  # for each layer
    print('** Actual replay buffer size: {}'.format(args.rmsize))
    agent = DDPG(nb_states, nb_actions, args)

    best_policy, best_reward = train(args.train_episode, agent, env, args.output, linear_quantization=args.linear_quantization, debug=args.debug)
    print('best_reward: ', best_reward)
    print('best_policy: ', best_policy)

