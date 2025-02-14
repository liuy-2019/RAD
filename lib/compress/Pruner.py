import torch
import numpy as np
from nni.compression.pruning import LevelPruner, L1NormPruner
from nni.compression.speedup import ModelSpeedup
import torch.nn.utils.prune as prune


class ModelPruner:
    def __init__(self, model, config_list, device, dummy_input=torch.rand((1, 3, 28, 28))):
        self.model = model
        self.config_list = config_list
        self.device = device
        self.dummy_input = dummy_input.to(device)

    def prune(self):
        self.pruner = L1NormPruner(self.model, self.config_list)
        _, mask = self.pruner.compress()

        self.pruner.unwrap_model()
        # dummy_input = torch.rand((1, 1, 28, 28)).to(self.device)

        model_speedup = ModelSpeedup(self.model, self.dummy_input, mask).speedup_model()

        return model_speedup

    def pytorch_prune(self, amount=0.9, remove=True):
        parameters_to_prune = [
            (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules())

        ]
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        if remove:
            # Freeze the pruned weights
            for module in filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules()):
                prune.remove(module, "weight")

        for name, param in self.model.named_parameters():
            if 'weight_orig' in name:  # Adjust based on your pruning method
                param.requires_grad = False
        return self.model

    def check_sparsity(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                print(
                    "Sparsity in {}.weight: {:.2f}%".format(
                        name, 100. * float(torch.sum(module.weight == 0))
                              / float(module.weight.nelement())
                    )
                )
            elif isinstance(module, torch.nn.Linear):
                print(
                    "Sparsity in {}.weight: {:.2f}%".format(
                        name, 100. * float(torch.sum(module.weight == 0))
                              / float(module.weight.nelement())
                    )
                )

# if __name__ == '__main__':
#     from model import adapter_block, prc_model

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     prune_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#     block = adapter_block(n_channels=256)
#     model = prc_model()
#     model.cuda()
#     block.cuda()
#     model.update(0, block)  # for ResNet-18 use get_resnet18().to(device)

#     checkpoint = torch.load('./checkpoint/checkpoint_resnet_20.th', map_location=device)
#     # 提取出模型的 state_dict
#     model_state_dict = checkpoint['state_dict']
#     model.load_state_dict(model_state_dict)

#     for level in prune_levels:
#         config_list = [{'sparsity': level, 'op_types': ['Conv2d']}]
#         pruner = ModelPruner(model, config_list, device, dummy_input=torch.rand((1, 3, 224, 224)).to(device))
#         compressed_model = pruner.prune() #  基于nni
#         # compressed_model = pruner.pytorch_prune()  # 基于pytorch
#         torch.save(compressed_model.state_dict(), f'checkpoint/pruned{level}.pth')

