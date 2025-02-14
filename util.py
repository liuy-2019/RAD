# from model import prc_model
# from model import Denoising
# import numpy as np

# #测试模型的参数量，以及插入那一层的参数量
# def test(net):
#     total_params = 0
#     for x in filter(lambda p: p.requires_grad, net.parameters()):
#         total_params += np.prod(x.data.numpy().shape)
#     print("Total number of params", total_params)
#     print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


# #计算数据的平均值
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# if __name__ == "__main__":
#     model = Denoising(64)
#     model = prc_model(num_classes=10)
#     test(model)