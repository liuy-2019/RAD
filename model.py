import torch
from torchvision.models.resnet import resnet50
from torch import nn
import torch.nn.functional as F
from model2 import ResNet18, ResNet50

# class adapter_block(nn.Module):
#     def __init__(self,n_channels, image_channels=1, use_bnorm=True, kernel_size=3, padding=1):
#         super(adapter_block, self).__init__()
#         layers = []

#         layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4*n_channels, kernel_size=kernel_size, padding=padding, bias=False))
#         layers.append(nn.BatchNorm2d(4*n_channels, eps=0.0001, momentum = 0.95))
#         layers.append(nn.ReLU(inplace=True))

#         layers.append(nn.Conv2d(in_channels=4*n_channels, out_channels=4*n_channels, kernel_size=kernel_size, padding=padding, bias=False))
#         layers.append(nn.BatchNorm2d(4*n_channels, eps=0.0001, momentum = 0.95))
#         layers.append(nn.ReLU(inplace=True))

#         layers.append(nn.Conv2d(in_channels=4*n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
#         layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
#         layers.append(nn.ReLU(inplace=True))

#         self.dncnn = nn.Sequential(*layers)
    
#     def forward(self, x):
#         y = x
#         out = self.dncnn(x)
#         return y+out
        

class prc_model(nn.Module):
    def __init__(self, num_classes=10):
        super(prc_model, self).__init__()
        # self.backbone = resnet50(pretrained=True)
        # self.backbone = resnet50(num_classes=num_classes)
        self.backbone = ResNet18(num_classes=num_classes)
        self.adapters = nn.ModuleList([nn.Identity() for _ in range(20)])
        # self.denoiser1 = Denoising(in_channels=256)
        # self.denoiser2 = Denoising(in_channels=512,feature_map_size=(4,4))
        # self.denoiser3 = Denoising(in_channels=1024)
        # self.denoiser4 = Denoising(in_channels=2048)

    def update(self, idx, block):
        self.adapters[idx] = block

    def remove(self, idx):
        self.adapters[idx] = nn.Identity()

    #resnet50 layer1 0-2 layer2 3-6 layer3 7-12 layer4 13-15
    # def forward(self, x):
    #     x = self.backbone.conv1(x)
    #     x = self.backbone.bn1(x)
    #     x = F.relu(x)
    #     # x = self.backbone.relu(x)
    #     # x = self.backbone.maxpool(x)

    #     for i, Bottleneck in enumerate(self.backbone.layer1):
    #         x = Bottleneck(x)
    #         #print(i,x.size())
    #         x = self.adapters[i](x)

    #     for i, Bottleneck in enumerate(self.backbone.layer2):
    #         x = Bottleneck(x)
    #         #print(i+3,x.size())
    #         x = self.adapters[i+3](x)

    #     for i, Bottleneck in enumerate(self.backbone.layer3):
    #         x = Bottleneck(x)
    #         #print(i+7,x.size())
    #         x = self.adapters[i+7](x)

    #     for i, Bottleneck in enumerate(self.backbone.layer4):
    #         x = Bottleneck(x)
    #         #print(i+13,x.size())
    #         x = self.adapters[i+13](x)

    #     # x = self.backbone.layer1(x)
    #     # x = self.adapters[0](x)
    #     # x = self.denoiser1(x)
    #     # x = self.backbone.layer2(x)
    #     # print("x feature",x.size())
    #     # x, alpha2 = self.denoiser2(x)
    #     # x = self.adapters[1](x)
    #     # x = self.backbone.layer3(x)
    #     # x = self.adapters[2](x)
    #     # # x = self.denoiser3(x)
    #     # x = self.backbone.layer4(x)
    #     # x = self.adapters[3](x)
    #     # x = self.denoiser4(x)

    #     # x = self.backbone.avgpool(x)
    #     x = F.avg_pool2d(x, 4)
    #     x = x.view(x.size(0), -1)
    #     # x = self.backbone.fc(x)
    #     x = self.backbone.linear(x)
    #     return x


    # resnet18 layer1 0-1 layer2 2-3 layer3 4-5 layer4 6-7
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x)

        for i, Bottleneck in enumerate(self.backbone.layer1):
            x = Bottleneck(x)
            # print(i,x.size())
            x = self.adapters[i](x)

        for i, Bottleneck in enumerate(self.backbone.layer2):
            x = Bottleneck(x)
            # print(i+2,x.size())
            x = self.adapters[i+2](x)

        for i, Bottleneck in enumerate(self.backbone.layer3):
            x = Bottleneck(x)
            # print(i+4,x.size())
            x = self.adapters[i+4](x)

        for i, Bottleneck in enumerate(self.backbone.layer4):
            x = Bottleneck(x)
            # print(i+6,x.size())
            x = self.adapters[i+6](x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        # x = self.backbone.fc(x)
        x = self.backbone.linear(x)
        return x


# 定义非局部操作
class NonLocalOp(nn.Module):
    def __init__(self, in_channels, embed=True, softmax=True):
        super(NonLocalOp, self).__init__()
        self.embed = embed
        self.softmax = softmax
        self.in_channels = in_channels

        if embed:
            # 定义 theta 和 phi 的 1x1 卷积
            self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
            self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, bias=False)
        else:
            # 如果不嵌入，则 theta 和 phi 直接使用输入 l
            self.theta = self.phi = None

    def forward(self, l):
        batch_size, C, H, W = l.shape

        # 根据 embed 是否开启来决定是否使用 theta 和 phi
        if self.embed:
            # print(f"L: {l.shape}")
            theta = self.theta(l)  # (B, C/2, H, W)
            phi = self.phi(l)  # (B, C/2, H, W)
            g = l  # g 和 l 一样 (B, C, H, W)
        else:
            theta, phi, g = l, l, l

        if C > H * W or self.softmax:
            # 使用 torch.einsum 进行张量乘积
            f = torch.einsum('bchw,bcxy->bhwxy', theta, phi)  # (B, H, W, H, W)
            if self.softmax:
                orig_shape = f.shape
                f = f.view(batch_size, H * W, H * W)  # (B, HW, HW)
                f = f / torch.sqrt(torch.tensor(C // 2, dtype=f.dtype).to(l.device))
                f = F.softmax(f, dim=-1)
                f = f.view(orig_shape)  # 恢复原始形状
            f = torch.einsum('bhwxy,bcxy->bchw', f, g)  # (B, C, H, W)
        else:
            f = torch.einsum('bchw,bchw->bc', phi, g)  # (B, C)
            f = torch.einsum('bc,bchw->bchw', f, theta)  # (B, C, H, W)

        if not self.softmax:
            f = f / (H * W)

        return f


# 定义去噪模块
class Denoising(nn.Module):
    def __init__(self, in_channels, embed=True, softmax=True, feature_map_size=(4,4)):
        super(Denoising, self).__init__()
        self.non_local_op = NonLocalOp(in_channels, embed=embed, softmax=softmax)

        # 定义 1x1 卷积层和批归一化
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

        # self.fc1 = nn.Conv2d(in_channels, in_channels, kernel_size=feature_map_size,stride=1, bias=False)
        # self.fc2 = nn.Linear(in_features=512, out_features=512)
        # self.fc2 = nn.Conv2d(in_channels, in_channels, kernel_size=feature_map_size,stride=1, bias=False)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=in_channels//4, out_channels=1, kernel_size=1)
        #self.fc = nn.Linear(1, 1)  # 输出两个类别的概率
        
        
    def forward(self, l):
        # 非局部操作
        f = self.non_local_op(l)

        # 1x1 卷积 + 批归一化
        f = self.conv(f)
        f = self.bn(f)

        #二分类任务
        # 卷积 -> ReLU激活函数 -> 池化
        x = F.relu(self.conv1(l))
        # print("x size after conv1",x.size())
        x = F.relu(self.conv2(x))
        # print("x size after conv1",x.size())
        alpha = F.relu(self.conv3(x))
        # print("x size after conv1",x.size())

        #alpha = x.view(x.size(0), -1)
        #print("alpha size",alpha.size())


        # 选择 alpha 的第一个类别
        # alpha = alpha[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # Reshape alpha to be able to broadcast to the shape of f and l
        # alpha = alpha.view(x.size(0), 1, 1, 2)  # shape becomes [256, 1, 1, 1]

        # print("alpha shape",alpha.size())
        # print("f shape",f.size())
        # print("l shape",l.size())

        # 残差连接
        #out = l + f * alpha
        out = torch.where(alpha > 0.5, l + f, l)

        # alpha = alpha.view(x.size(0), -1)
        # alpha = self.fc(alpha)
        # alpha = torch.sigmoid(alpha)

        # print("alpha = ",alpha)
        return out #.view(alpha.size(0),-1)


# if __name__ == "__main__":
#     # block = adapter_block(n_channels=256)
#     model = prc_model()
#     # model.update(0,block)
#     print(model)

#     x = torch.rand((2, 3, 32, 32))
#     y = model(x)
#     print(y.size())