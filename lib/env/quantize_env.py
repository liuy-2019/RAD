# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar

from lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from lib.utils.data_utils import get_split_train_dataset
from lib.utils.quantize_utils import quantize_model, kmeans_update_model, insert_denoiser, remove_denoiser
from lib.compress.robustbench import load_cifar10

from model import prc_model
from model import Denoising


from autoattack import AutoAttack
import foolbox as fb
from lib.utils.utils import AverageMeter

from lib.utils.adv import trades_loss

# todo 
# 1.现在的策略鲁棒性提升有多少
# 2.对抗性攻击作为reward和微调
# 3.插入的特征维度应该多一点


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)



class QuantizeEnv:
    def __init__(self, model, pretrained_model, model_arch, data, data_root, compress_ratio, args, n_data_worker=16,
                 batch_size=256, float_bit=32, is_model_pruned=False):
        # default setting
        self.quantizable_layer_types = [nn.Conv2d, nn.Linear]

        # save options
        self.model = model
        # self.model_for_measure = deepcopy(model)
        # state_dict = model.state_dict()
        # new_model = type(model)()  # 创建一个新的模型实例
        # new_model.load_state_dict(deepcopy(state_dict))
        self.model_for_measure = deepcopy(model_arch)

        self.cur_ind = 0
        self.strategy = []  # quantization strategy

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.pretrained_model = pretrained_model
        self.n_data_worker = n_data_worker
        self.batch_size = batch_size
        self.data_type = data
        self.data_root = data_root
        self.compress_ratio = compress_ratio
        self.is_model_pruned = is_model_pruned
        self.val_size = args.val_size
        self.train_size = args.train_size
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.float_bit = float_bit * 1.
        self.last_action = self.max_bit

        self.is_inception = args.arch.startswith('inception')
        self.is_imagenet = ('imagenet' in data)
        self.use_top5 = args.use_top5

        # sanity check
        assert self.compress_ratio > self.min_bit * 1. / self.float_bit, \
            'Error! You can make achieve compress_ratio smaller than min_bit!'

        # init reward
        self.best_reward = -math.inf

        # prepare data
        self._init_data()

        # build indexs
        self._build_index()
        # self._get_weight_size()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.org_acc = self._validate(self.val_loader, self.model)
        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # restore weight
        self.reset()
        # print('=> original acc: {:.3f}% on split dataset(train: %7d, val: %7d )'.format(self.org_acc,
        #                                                                                 self.train_size, self.val_size))
        # print('=> original #param: {:.4f}, model size: {:.4f} MB'.format(sum(self.wsize_list) * 1. / 1e6,
        #                                                                  sum(self.wsize_list) * self.float_bit / 8e6))

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        # 伪剪枝并获取相应的统计数据。真正的剪枝要等到所有伪剪枝结束后才进行

        action = self._action_wall(action)  # percentage to preserve

        self.strategy.append(action)  # save action to strategy

        # all the actions are made
        if self._is_final_layer():
            # self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            w_size = self._cur_weight()
            w_size_ratio = self._cur_weight() / self._org_weight() #压缩率：插入的参数量/原参数量

            # centroid_label_dict = quantize_model(self.model, self.quantizable_idx, self.strategy,
            #                                      mode='cpu', quantize_bias=False, centroids_init='k-means++',
            #                                      is_pruned=self.is_model_pruned, max_iter=3)
            insert_denoiser(self.model, self.quantizable_idx, self.strategy, self.inchannel_list)

            if self.finetune_flag:
                self._kmeans_finetune2(self.train_loader, self.model, self.finetune_epoch, self.batch_size)
                train_acc = self._kmeans_finetune(self.train_loader, self.model, epochs=self.finetune_epoch)

                acc = self._validate(self.val_loader, self.model)
                robust_acc = self._test_robust2(self.model)
                acc = (acc + robust_acc)/2
            else:
                acc = self._validate(self.val_loader, self.model)

            # 这里决定计算reward的时候是否要考虑w_size_ratio
            reward = self.reward(acc, w_size_ratio)
            #reward = self.reward(acc)

            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, w_size_ratio))

            obs = self.layer_embedding[self.cur_ind, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_embedding[self.cur_ind][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    # for quantization
    def reward(self, acc, w_size_ratio=None):
        if w_size_ratio is not None:
            return (acc - self.org_acc + 1. / w_size_ratio) * 0.1
        return (acc - self.org_acc) * 0.1

    def reset(self):
        # restore env by loading the pretrained model
        remove_denoiser(self.model, self.quantizable_idx, self.strategy)
        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        print("current index id:",self.cur_ind)
        print("quantizable_idx len",len(self.quantizable_idx))
        return self.cur_ind == len(self.quantizable_idx) - 1

    # def _final_action_wall(self): #根据压缩率调整bit位，保证满足压缩率
    #     target = self.compress_ratio * self._org_weight()
    #     min_weight = 0
    #     for i, n_bit in enumerate(self.strategy):
    #         min_weight += self.wsize_list[i] * self.min_bit
    #     while min_weight < self._cur_weight() and target < self._cur_weight():
    #         for i, n_bit in enumerate(reversed(self.strategy)):
    #             if n_bit > self.min_bit:
    #                 self.strategy[-(i+1)] -= 1
    #             if target >= self._cur_weight():
    #                 break
    #     print('=> Final action list: {}'.format(self.strategy))

    # def _action_wall(self, action):
    #     assert len(self.strategy) == self.cur_ind
    #     # limit the action to certain range
    #     action = float(action)
    #     min_bit, max_bit = self.bound_list[self.cur_ind]
    #     lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
    #     action = (rbound - lbound) * action + lbound
    #     action = int(np.round(action, 0))
    #     self.last_action = action
    #     return action  # not constrained here

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind, 'Strategy length: {} cur_ind: {}'.format(len(self.strategy), self.cur_ind)
        action = float(action)

        if action > 0.5:
            action = 1
        else:
            action = 0

        self.last_action = action
        return action

    def _cur_weight(self): #todo 改掉这个方式
        # cur_weight = 0.
        # # quantized
        # for i, n_bit in enumerate(self.strategy):
        #     cur_weight += n_bit * self.wsize_list[i]

        cur_weight = 0 #总参数量
        for i, insert in enumerate(self.strategy):
            if insert == 1:
                total_params = 0 #每个插入模块的参数量
                net = Denoising(self.inchannel_list[i])
                for x in filter(lambda p: p.requires_grad, net.parameters()):
                    total_params += np.prod(x.data.numpy().shape)
                cur_weight += total_params
        print("toal weight of adapted denoisers",cur_weight)
        return cur_weight

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_bitops - self._cur_bitops()
        return reduced

    def _org_weight(self):
        net = prc_model(num_classes=10)
        total_params = 0
        for x in filter(lambda p: p.requires_grad, net.parameters()):
            total_params += np.prod(x.data.numpy().shape)
        # org_weight = 0.
        # org_weight += sum(self.wsize_list) * self.float_bit
        print("total weight of original model",total_params)
        return total_params

    def _init_data(self):
        self.train_loader, self.val_loader, n_class = get_split_train_dataset(
            self.data_type, self.batch_size, self.n_data_worker, data_root=self.data_root,
            val_size=self.val_size, train_size=self.train_size, for_inception=self.is_inception)

    def _build_index(self):
        # self.quantizable_idx = []
        # self.layer_type_list = []
        # self.bound_list = []
        # for i, m in enumerate(self.model.modules()):
        #     if type(m) in self.quantizable_layer_types:
        #         self.quantizable_idx.append(i)
        #         self.layer_type_list.append(type(m))
        #         self.bound_list.append((self.min_bit, self.max_bit))
        # print('=> Final bound list: {}'.format(self.bound_list))

        self.quantizable_idx = [] #表示可以插入的层的index 原则是可以在每一个Bottleneck之后插入
        # self.layer_type_list = [] #表示被插入的层的类型
        self.inchannel_list = [] #表示被插入层的inchannel数
        #print(self.model.state_dict().keys())
        
        # resnet50
        # for i, m in enumerate(self.model.modules()):
        #     # print(i,m)
        #     class_name = m.__class__.__name__
        #     if class_name == 'Bottleneck':
        #         self.quantizable_idx.append(i+1)

        # self.inchannel_list=[256, 256, 256, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 2048, 2048, 2048]


        #resnet18
        for i, m in enumerate(self.model.modules()):
            # print(i,m)
            class_name = m.__class__.__name__
            if class_name == 'BasicBlock':
                self.quantizable_idx.append(i+1)
        
        self.inchannel_list=[64, 64, 128, 128, 256, 256, 512, 512]

        print('=> Final quantizable_idx list: {}'.format(self.quantizable_idx))
        print('=> Final inchannel_list: {}'.format(self.inchannel_list))
        # print('=> Final layer_type_list: {}'.format(self.layer_type_list))
        
    # def _get_weight_size(self):
    #     # get the param size for each layers to prune, size expressed in number of params
    #     self.wsize_list = []
    #     for i, m in enumerate(self.model.modules()):
    #         if i in self.quantizable_idx:
    #             if not self.is_model_pruned:
    #                 self.wsize_list.append(m.weight.data.numel())
    #             else:  # the model is pruned, only consider non-zeros items
    #                 self.wsize_list.append(torch.sum(m.weight.data.ne(0)))
    #     self.wsize_dict = {i: s for i, s in zip(self.quantizable_idx, self.wsize_list)}

    def _get_latency_list(self):
        # use simulator to get the latency
        raise NotImplementedError

    def _get_energy_list(self):
        # use simulator to get the energy
        raise NotImplementedError

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        # print("imagenet data",self.is_imagenet)
        # if self.is_imagenet:
        #     measure_model(self.model_for_measure, 224, 224)
        # else:
        #     measure_model(self.model_for_measure, 32, 32)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        # print("+++++++++++++++++ module list +++++++++++++++++++++++++++",module_list)

        real_module_list = list(self.model.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            # m = m.bn3
            this_state = []
            # this_state.append([m.in_w*m.in_h])  # input feature_map_size
            this_state.append(ind)
            layer_embedding.append(np.hstack(this_state))
            this_state.append([m.in_channels])  # in channels
            this_state.append([m.out_channels]) # out channels

            # if type(m) == nn.Conv2d or type(m) == nn.Conv2d:
            #     this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
            #     this_state.append([m.in_channels])  # in channels
            #     this_state.append([m.out_channels]) # out channels
            #     this_state.append([m.stride[0]])  # stride
            #     this_state.append([m.kernel_size[0]])  # kernel size
            #     this_state.append([np.prod(m.weight.size())])  # weight size
            #     this_state.append([m.in_w*m.in_h])  # input feature_map_size
            # elif type(m) == nn.Linear or type(m) == nn.Linear:
            #     this_state.append([0.])  # layer type, 0 for fc
            #     this_state.append([m.in_features])  # in channels
            #     this_state.append([m.out_features])  # out channels
            #     this_state.append([0.])  # stride
            #     this_state.append([1.])  # kernel size
            #     this_state.append([np.prod(m.weight.size())])  # weight size
            #     this_state.append([m.in_w*m.in_h])  # input feature_map_size

            # this_state.append([i])  # index
            # this_state.append([4.])  # bits
            # layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding


    # def new_lr(optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = lr

    def _cosine_schedule(self, optimizer, epochs, warmup_lr=0.1, warmup_epochs=0, lr=0.1):
        def set_lr(epoch, lr=lr, epochs=epochs):
            if epoch < warmup_epochs:
                a = warmup_lr
            else:
                epoch = epoch - warmup_epochs
                a = lr * 0.5 * (1 + np.cos((epoch - 1) / epochs * np.pi))

            # new_lr(optimizer, a)
            for param_group in optimizer.param_groups:
                param_group["lr"] = a

        return set_lr
    


    def _kmeans_finetune2(self, train_loader, model, epochs, batch_size):

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

        lr_policy = self._cosine_schedule(optimizer, epochs)

        for epoch in range(0, epochs):
            lr_policy(epoch)  # adjust learning rate

            print(
                " ->->->->->->->->->-> One epoch with Adversarial training (TRADES) <-<-<-<-<-<-<-<-<-<-"
            )

            batch_time = AverageMeter("Time", ":6.3f")
            data_time = AverageMeter("Data", ":6.3f")
            losses = AverageMeter("Loss", ":.4f")
            top1 = AverageMeter("Acc_1", ":6.2f")
            top5 = AverageMeter("Acc_5", ":6.2f")

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch),
            )

            model.train()
            end = time.time()
            device = torch.device("cuda")

            for i, data in enumerate(train_loader):
                images, target = data[0].to(device), data[1].to(device)

                # basic properties of training data
                if i == 0:
                    print(
                        images.shape,
                        target.shape,
                        f"Batch_size from args: {batch_size}",
                        "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
                    )
                    print(f"Training images range: {[torch.min(images), torch.max(images)]}")

                output = model(images)

                # calculate robust loss
                loss = trades_loss(
                    model=model,
                    x_natural=images,
                    y=target,
                    device=device,
                    optimizer=optimizer,
                    step_size=0.0078,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=6.0,
                    clip_min=0,
                    clip_max=1,
                    distance="l_inf",
                )

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                progress.display(i)


    def _kmeans_train2(self, train_loader, model, epochs, batch_size, val_loader):

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=0.0001,
        )

        lr_policy = self._cosine_schedule(optimizer, epochs)
        best_acc = 0

        for epoch in range(0, epochs):
            lr_policy(epoch)  # adjust learning rate

            print(
                " ->->->->->->->->->-> One epoch with Adversarial training (TRADES) <-<-<-<-<-<-<-<-<-<-"
            )

            batch_time = AverageMeter("Time", ":6.3f")
            data_time = AverageMeter("Data", ":6.3f")
            losses = AverageMeter("Loss", ":.4f")
            top1 = AverageMeter("Acc_1", ":6.2f")
            top5 = AverageMeter("Acc_5", ":6.2f")

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch),
            )

            model.train()
            end = time.time()
            device = torch.device("cuda")

            for i, data in enumerate(train_loader):
                images, target = data[0].to(device), data[1].to(device)

                # basic properties of training data
                if i == 0:
                    print(
                        images.shape,
                        target.shape,
                        f"Batch_size from args: {batch_size}",
                        "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
                    )
                    print(f"Training images range: {[torch.min(images), torch.max(images)]}")

                output = model(images)

                # calculate robust loss
                loss = trades_loss(
                    model=model,
                    x_natural=images,
                    y=target,
                    device=device,
                    optimizer=optimizer,
                    step_size=0.0078,
                    epsilon=0.031,
                    perturb_steps=10,
                    beta=6.0,
                    clip_min=0,
                    clip_max=1,
                    distance="l_inf",
                )

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                progress.display(i)


            # 在验证集上评估模型
            val_accuracy = self._validate(val_loader, model)
            
            # 如果当前epoch的准确率高于之前的最高准确率，则更新并保存模型
            if val_accuracy > best_acc:
                print(f"Epoch {epoch + 1}: New best accuracy: {val_accuracy:.4f}")
                best_accuracy = val_accuracy
                
                # 创建一个字典来保存模型的状态和其他信息
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,
                }
                
                # 保存到文件
                torch.save(checkpoint, "resnet19_NWPU_adv_train2.pt")
        


    def _kmeans_train(self, train_loader, model, val_loader, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()
        bar = Bar('_kmeans_finetune:', max=len(train_loader))
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                input_var, target_var = inputs.cuda(), targets.cuda()
                #print(f"Validation data range: min={inputs.min().item()}, max={inputs.max().item()}")

                #加入对抗性攻击 FGSM
                # new_model = deepcopy(model)
                # new_model.eval()
                # fmodel = fb.PyTorchModel(new_model, bounds=(-3, 3)) #0,1
                # attack = fb.attacks.FGSM()
                # raw_advs, x_fgm, success = attack(fmodel, input_var, criterion=fb.criteria.Misclassification(target_var),epsilons=[8/255])
                # input_var = x_fgm[0]


                #加入对抗性攻击 PGD
                new_model = deepcopy(model)
                new_model.eval()
                fmodel = fb.PyTorchModel(new_model, bounds=(-3, 6)) #0,1
                attack = fb.attacks.LinfPGD()
                _, x_pgd, _  = attack(fmodel, input_var ,criterion=fb.criteria.Misclassification(target_var), epsilons=[8/(255)])
                input_var = x_pgd[0]
            

                #加入对抗性攻击AutoAttack
                # device = torch.device("cuda")
                # adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                # adversary.apgd.n_restarts = 1
                # x_aa = adversary.run_standard_evaluation(input_var, target_var)
                # input_var = x_aa

                # Pre-generate adversarial examples once per epoch and mix with clean samples
                # adversarial_update_freq=2
                # if epoch % adversarial_update_freq == 0:
                #     adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                #     adversary.apgd.n_restarts = 1
                #     x_aa = adversary.run_standard_evaluation(input_var, target_var)
                #     # Mix clean and adversarial examples
                #     mixed_inputs = torch.cat([input_var, x_aa], dim=0)
                #     mixed_targets = torch.cat([target_var, target_var], dim=0)
                # else:
                #     mixed_inputs, mixed_targets = input_var, target_var

                #噪声扰动
                # noise = torch.randn_like(inputs, device='cuda') * 0.25
                # input_var += noise

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()

                # do SGD step
                self.optimizer.step()

                # kmeans_update_model(model, self.quantizable_idx, centroid_label_dict, free_high_bit=True)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()


            # 在验证集上评估模型
            val_accuracy = self._validate(val_loader, model)
            
            # 如果当前epoch的准确率高于之前的最高准确率，则更新并保存模型
            if val_accuracy > best_acc:
                print(f"Epoch {epoch + 1}: New best accuracy: {val_accuracy:.4f}")
                best_accuracy = val_accuracy
                
                # 创建一个字典来保存模型的状态和其他信息
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,
                }
                
                # 保存到文件
                torch.save(checkpoint, "purl_resnet18_NWPU.pt")

        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return best_acc



    def _kmeans_finetune(self, train_loader, model, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_acc = 0.

        # switch to train mode
        model.train()
        end = time.time()
        t1 = time.time()
        bar = Bar('_kmeans_finetune:', max=len(train_loader))
        for epoch in range(epochs):
            for i, (inputs, targets) in enumerate(train_loader):
                input_var, target_var = inputs.cuda(), targets.cuda()
                #print(f"Validation data range: min={inputs.min().item()}, max={inputs.max().item()}")

                #加入对抗性攻击 FGSM
                # new_model = deepcopy(model)
                # new_model.eval()
                # fmodel = fb.PyTorchModel(new_model, bounds=(-3, 3)) #0,1
                # attack = fb.attacks.FGSM()
                # raw_advs, x_fgm, success = attack(fmodel, input_var, criterion=fb.criteria.Misclassification(target_var),epsilons=[8/255])
                # input_var = x_fgm[0]


                # #加入对抗性攻击 PGD
                # new_model = deepcopy(model)
                # new_model.eval()
                # fmodel = fb.PyTorchModel(new_model, bounds=(-3, 3)) #0,1
                # attack = fb.attacks.LinfPGD()
                # _, x_pgd, _  = attack(fmodel, input_var ,criterion=fb.criteria.Misclassification(target_var), epsilons=[8/(255)])
                # input_var = x_pgd[0]
            

                #加入对抗性攻击AutoAttack
                # device = torch.device("cuda")
                # adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                # adversary.apgd.n_restarts = 1
                # x_aa = adversary.run_standard_evaluation(input_var, target_var)
                # input_var = x_aa

                # Pre-generate adversarial examples once per epoch and mix with clean samples
                # adversarial_update_freq=2
                # if epoch % adversarial_update_freq == 0:
                #     adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
                #     adversary.apgd.n_restarts = 1
                #     x_aa = adversary.run_standard_evaluation(input_var, target_var)
                #     # Mix clean and adversarial examples
                #     mixed_inputs = torch.cat([input_var, x_aa], dim=0)
                #     mixed_targets = torch.cat([target_var, target_var], dim=0)
                # else:
                #     mixed_inputs, mixed_targets = input_var, target_var

                #噪声扰动
                # noise = torch.randn_like(inputs, device='cuda') * 0.25
                # input_var += noise

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                output = model(input_var)
                loss = self.criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # compute gradient
                self.optimizer.zero_grad()
                loss.backward()

                # do SGD step
                self.optimizer.step()

                # kmeans_update_model(model, self.quantizable_idx, centroid_label_dict, free_high_bit=True)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.val,
                            bt=batch_time.val,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            top1=top1.avg,
                            top5=top5.avg,
                        )
                    bar.next()
            bar.finish()

            if self.use_top5:
                if top5.avg > best_acc:
                    best_acc = top5.avg
            else:
                if top1.avg > best_acc:
                    best_acc = top1.avg
            self.adjust_learning_rate()
        t2 = time.time()
        if verbose:
            print('* Test loss: %.3f  top1: %.3f  top5: %.3f  time: %.3f' % (losses.avg, top1.avg, top5.avg, t2-t1))
        return best_acc

    def _validate(self, val_loader, model, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

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
                loss = self.criterion(output, target_var)

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
        if self.use_top5:
            return top5.avg
        else:
            return top1.avg


    def _test_robust(self,test_loader, model, criterion):
        """
        Run evaluation
        """
        
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        # top_fgsm = AverageMeter()
        # top_pgd = AverageMeter()
        top_aa = AverageMeter()
        # switch to evaluate mode
        model.eval()

        end = time.time()

        # clean_model = resnet50(num_classes=10).cuda()
        # clean_model.eval()

        # with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            adversary = AutoAttack(model, norm='Linf', eps=8/255)
            x_aa = adversary.run_standard_evaluation(input_var, target_var)

            # fmodel = fb.PyTorchModel(model, bounds=(0, 1))
            # fmodel = fb.PyTorchModel(model, bounds=(-3, 3)) #
            # attack = fb.attacks.FGSM()
            # raw_advs, x_fgm, success = attack(fmodel, input_var, criterion=fb.criteria.Misclassification(target_var),epsilons=[8/255])


            # attack = fb.attacks.LinfPGD()
            #_, x_pgd, _  = attack(fmodel, input_var ,criterion=fb.criteria.Misclassification(target_var), epsilons=[8/(255)])
            output = model(input_var)
            
            # output_fgm = model(x_fgm[0])
            # output_pgd = model(x_pgd[0]) 
            output_aa = model(x_aa)
            
            output = output.float()
            # output_fgm = output_fgm.float()
            # output_pgd = output_pgd.float()
            output_aa = output_aa.float()

            prec1 = accuracy(output.data, target)[0]
            # prec1_fgm = accuracy(output_fgm.data, target)[0]
            # prec1_pgd = accuracy(output_pgd.data, target)[0]
            prec1_aa = accuracy(output_aa.data, target)[0]

            top1.update(prec1.item(), input.size(0))
            # top_fgsm.update(prec1_fgm.item(), input.size(0))
            # top_pgd.update(prec1_pgd.item(), input.size(0))
            top_aa.update(prec1_aa.item(), input.size(0))

            loss = criterion(output, target_var)
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0: #每10轮打印一次
                print('_test_robust: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        #'Precfgsm@1 {top_fgsm.val:.3f} ({top_fgsm.avg:.3f})\t'
                        #'Precpgd@1 {top_pgd.val:.3f} ({top_pgd.avg:.3f})\t'
                        'Precaa@1 {top_aa.val:.3f} ({top_aa.avg:.3f})\t'.format(
                            i, len(test_loader), batch_time=batch_time, loss=losses,
                            top1=top1, 
                            #top_pgd=top_pgd, top_fgsm=top_fgsm, 
                            top_aa=top_aa))
        print(' * Prec@1 {:.3f} * Precfgsm@1 {:.3f} * Precpgd@1 {:.3f} * Precaa@1 {:.3f}'
        .format(top1.avg, 
                #top_fgsm.avg, top_pgd.avg, 
                top_aa.avg))
        #return ' * Prec@1 {:.3f} * Precfgsm@1 {:.3f} * Precpgd@1 {:.3f} * Precaa@1 {:.3f}'.format(top1.avg, top_fgsm.avg, top_pgd.avg, top_aa.avg)
        #return (top_fgsm.avg + top_pgd.avg + top_aa.avg)/3
        # return (top_fgsm.avg + top_pgd.avg)/2
        return top_aa.avg



    def _test_robust2(self, model):

        device = torch.device("cuda")
        x_test, y_test = load_cifar10(n_examples=50)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
        adversary.apgd.n_restarts = 1
        auto_robust_acc = adversary.run_standard_evaluation(x_test, y_test, robust_acc=True)
        return auto_robust_acc