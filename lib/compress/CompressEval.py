import torch
from lib.compress.TrainingUtil import *
import pandas as pd
from lib.compress import Quantizer
from lib.compress import Pruner
import os
#from model import adapter_block
#from model import prc_model
import copy


class CompressEval():
    def __init__(self,device,train_loader, test_loader, pgd, quantization_levels, quantization_methods, pruning_levels, pruning_methods, file_name, model_directory):
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pgd = pgd
        self.quantization_levels = quantization_levels
        self.quantization_methods = quantization_methods
        self.pruning_levels = pruning_levels
        self.pruning_methods = pruning_methods
        self.file_name = file_name
        self.model_directory = model_directory
        self.results = {'model_name':[],'compression_method':[],'sparsity_level':[],'test':[],'robustness':[]}

    def eval(self,model):
        '''
        Evaluate model by test and robustness, and save results to self.results dictionary
        '''
        test = epoch(self.test_loader, model)[0]
        robustness = epoch_adversarial(self.test_loader, model, self.pgd)[0]
        return test, robustness

    def save_results(self):
        '''Save results as a dataframe in a pickle file'''
        results_dataframe = pd.DataFrame(self.results)
        results_dataframe.to_pickle(f'./{self.file_name}.pkl')
        


    def get_results_dataframe(self):
        '''return results'''
        return pd.DataFrame(self.results)
    
    def clone_model(self,model):
        clone = type(model)()
        clone.load_state_dict(model.state_dict())
        clone.to(self.device)
        return clone
    
    def single_compress(self, model,model_name, compression_method, sparsity_level):
        '''Match compression_method name with the corresponding compression'''
        wandb_run_name = f'{model_name}_{compression_method}_{sparsity_level}'
        if compression_method == "none" and sparsity_level == "none":
            compressed_model = self.clone_model(model)
        elif compression_method == "ptq":
            compressed_model = self.clone_model(model)
            compressed_model = self.quantizer.quantize(compressed_model, robust_model=False, QAT=False, fine_tune=False, run_name=wandb_run_name)
        elif compression_method == "ptqNFT":
            compressed_model = self.clone_model(model)
            compressed_model = self.quantizer.quantize(compressed_model, robust_model=False, QAT=False, fine_tune=True, run_name=wandb_run_name)
        elif compression_method == "ptqRFT":
            compressed_model = self.clone_model(model)
            compressed_model = self.quantizer.quantize(compressed_model, robust_model=True, QAT=False, fine_tune=True, run_name=wandb_run_name)
        elif compression_method == "QATrobust":
            compressed_model = self.clone_model(model)
            compressed_model = self.quantizer.quantize(compressed_model, robust_model=True, QAT=True, run_name=wandb_run_name)
        elif compression_method == "prune":
            # compressed_model = self.clone_model(model)
            # compressed_model = self.pruner.prune()
            compressed_model = self.pruner.pytorch_prune(amount=sparsity_level)
        else:
            raise ValueError(f'compression_method "{compression_method}" not valid')
        return compressed_model

    def compress_eval_save(self,model,model_name,compression_method, sparsity_level):
        '''Compresses, evaluates and saves given model with compression_method and sparisity_level. Model_name is used for naming results'''
        #compress
        compressed_model = self.single_compress(model,model_name, compression_method, sparsity_level)

        #evaluate
        test, robustness = self.eval(compressed_model)

        #add to results
        self.results['model_name'].append(model_name)
        self.results['compression_method'].append(compression_method)
        self.results['sparsity_level'].append(sparsity_level)
        self.results['test'].append(test)
        self.results['robustness'].append(robustness)

        #save model
        torch.save(compressed_model.state_dict(), f'{self.model_directory}/{model_name}_{compression_method}_{sparsity_level}.pt')

    def multi_compress(self, model, model_name):
        '''
        Compresses given models with all sparsity levels and methods.
        Models are saved in model_directory
        '''

        #eval baseline(no compression applied)
        self.compress_eval_save(model,model_name,"none","none")

        #compress and eval for all levels and methods
        for level in self.quantization_levels:
            self.quantizer = Quantizer.quantizer(self.train_loader,self.test_loader, level, self.device, self.pgd, self.wandb_project_name)
            for method in self.quantization_methods:
                self.compress_eval_save(model,model_name,method,level)
        for level in self.pruning_levels:
            config_list = [{'sparsity': level, 'op_types': ['Conv2d']}]
            self.pruner = Pruner.ModelPruner(model, config_list, self.device)
            for method in self.pruning_methods:
                self.compress_eval_save(model,model_name,method,level)

        self.save_results()

    def single_prune_compress(self, model, model_name, pruning_level, pruning_methods):
        config_list = [{'sparsity': pruning_level, 'op_types': ['Conv2d']}]
        self.pruner = Pruner.ModelPruner(model, config_list, self.device)
        #compress
        compressed_model = self.single_compress(model, model_name, pruning_methods, pruning_level)
        return compressed_model
    
    def single_quant_compress(self, model, model_name, quant_level, quant_method):
        # data_loader = CIFAR10DataLoader() #FASHIONMNISTDataLoader() #For cifar10 use CIFAR10DataLoader()
        # self.train_loader = data_loader.train_loader
        # self.test_loader = data_loader.test_loader

        self.quantizer = Quantizer.quantizer(self.train_loader,self.test_loader, quant_level, self.device, self.pgd, self.wandb_project_name)
        # self.compress_eval_save(model,model_name,quant_method,quant_level)
        compressed_model = self.single_compress(model, model_name, quant_method, quant_level)
        return compressed_model