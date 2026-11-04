"""This file contains the model definition of TA-TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from einops import rearrange
from modeling.entropy.video_model import Compressive_bottleneck
from modeling.modules.blocks import GVC1DDecoder, GVC1DEncoder
from omegaconf import OmegaConf
from modeling.modules.base_model import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from modeling.modules.long_term_memory import long_term_feature_extractor
from torch.cuda.amp import autocast

class DiagonalGaussianDistribution(object):
    @autocast(enabled=False)
    def __init__(self, parameters, deterministic=False):
        """Initializes a Gaussian distribution instance given the parameters.

        Args:
            parameters (torch.Tensor): The parameters for the Gaussian distribution. It is expected
                to be in shape [B, 2 * C, *], where B is batch size, and C is the embedding dimension.
                First C channels are used for mean and last C are used for logvar in the Gaussian distribution.
            deterministic (bool): Whether to use deterministic sampling. When it is true, the sampling results
                is purely based on mean (i.e., std = 0).
        """
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters.float(), 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    @autocast(enabled=False)
    def sample(self):
        x = self.mean.float() + self.std.float() * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    @autocast(enabled=False)
    def mode(self):
        return self.mean

    @autocast(enabled=False)
    def kl(self):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            return 0.5 * torch.sum(torch.pow(self.mean.float(), 2)
                                    + self.var.float() - 1.0 - self.logvar.float(),
                                    dim=[1, 2])

class GVC1D(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        super().__init__()
        self.config = config
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.encoder = GVC1DEncoder(config)
        self.decoder = GVC1DDecoder(config)
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        if config.model.hyperprior.get('use_hyperprior', False):
            self.codec_model = Compressive_bottleneck(N=config.model.hyperprior.get('N', 16),
                                                        z_channel=config.model.hyperprior.get('z_channel', 8),
                                                        y_tokens_num=self.num_latent_tokens,
                                                        g_ch_src=config.model.vq_model.get('token_size', 128),
                                                        memory_num=config.model.memory.memory_num if config.model.memory.get('use_memory', False) else 0,
                                                        memory_dim=config.model.memory.memory_dim if config.model.memory.get('use_memory', False) else 0,
            )
        
        if config.model.memory.get('use_memory', False):
            memory_dim = config.model.memory.memory_dim
            total_memory_num = config.model.memory.total_memory_num
            memory_layer_nums = config.model.memory.get('memory_layers', 8)
            self.long_term_feature_extractor = long_term_feature_extractor(input_size=config.model.vq_model.get('token_size', 128), output_size=memory_dim, n_embd=memory_dim, layer_nums=memory_layer_nums, hidden_token_num=total_memory_num,query_num=config.model.memory.memory_num)
        
        training_module=config.model.get('training_module', None)
        if training_module == 'frame_condition' or training_module == 'codec_module':
            self.latent_tokens.requires_grad_(False)
        elif training_module == 'global_atten_module':
            self.latent_tokens.requires_grad_(False)
            self.codec_model.eval()
            self.codec_model.requires_grad_(False)
        elif training_module == 'memory_module':
            self.latent_tokens.requires_grad_(False)
            for name, param in self.codec_model.named_parameters():
                if not name.startswith('feature_extractor.long_term_feature_positional_embedding') and not name.startswith('feature_extractor.long_term_feature_projection'):
                    param.requires_grad = False

        self.dpb = {
            'ref_frame': None,
            'ref_feature': None,
            'long_term_mem': None,
        }
        self.frame_idx = 1
        self.quantize = DiagonalGaussianDistribution
        self.apply(self._init_weights)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x,cond=None, fa_idx=0, long_term_feature=None):
        z, stack_shape  = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens, cond=cond, fa_idx=fa_idx, long_term_feature=long_term_feature)
        if self.config.model.hyperprior.get('use_hyperprior', False):
            z_quantized, _ = torch.chunk(z, 2, dim=1)
            kl_result_dict = {}
        else:
            posteriors = self.quantize(z)
            z_quantized = posteriors.sample()
            kl_result_dict = posteriors
        return z_quantized, stack_shape, kl_result_dict
    
    def decode(self, z_quantized, cond=None, fa_idx=0, long_term_feature=None, stack_shape=None):
        decoded, feature = self.decoder(z_quantized, cond, fa_idx=fa_idx, long_term_feature=long_term_feature, stack_shape=stack_shape)
        return decoded, feature

    def forward_one_frame(self, x, dpb, qp, fa_idx=0):
        # get long_term_feature
        B, _, H, W = x.size()
        if self.config.model.memory.get('use_memory', False):
            if dpb['long_term_mem'] is None:
                nums_memory = B * H *W //(256*256)
                long_term_mem=self.long_term_feature_extractor.get_init_hidden_tokens(nums_memory)
            else:
                long_term_mem=dpb['long_term_mem']
            long_term_feature = self.long_term_feature_extractor.get_feature(long_term_mem)
        else:
            long_term_feature = None
            long_term_mem = None
        # Tatitok encode
        x, stack_shape, kl_result_dict = self.encode(x, cond=dpb, fa_idx=fa_idx, long_term_feature=long_term_feature)
        # ------------codec part------------
        if self.config.model.hyperprior.get('use_hyperprior', False):
            x_hat, codec_dic = self.codec_model(x, dpb, qp, H, W, fa_idx, long_term_feature, stack_shape=stack_shape)
        else:
            x_hat = x
            codec_dic = {}
        if self.config.model.memory.get('use_memory', False):
            long_term_mem = self.long_term_feature_extractor.update_feature(x_hat.squeeze(2).permute(0,2,1), long_term_mem)
        # ------------codec part------------
        x_hat,feature = self.decode(x_hat, cond=dpb, fa_idx=fa_idx, long_term_feature=long_term_feature, stack_shape=stack_shape)
        codec_dic['kl_result_dict'] = kl_result_dict
        codec_dic['x_hat'] = x_hat
        codec_dic['dpb'] = {
            "ref_frame": x_hat,
            "ref_feature": feature,
            "long_term_mem": long_term_mem,
        }
        return codec_dic

    def customize_task(self, task):
        with torch.no_grad():
            if task == 'c0t2':
                self.codec_model.feature_adaptor_p[2].load_state_dict(self.codec_model.feature_adaptor_p[0].state_dict())
                self.encoder.adapter_p[2].load_state_dict(self.encoder.adapter_p[0].state_dict())
                self.decoder.adapter_p[2].load_state_dict(self.decoder.adapter_p[0].state_dict())
            elif task == 'c2t1':
                self.codec_model.feature_adaptor_p[1].load_state_dict(self.codec_model.feature_adaptor_p[2].state_dict())
                self.encoder.adapter_p[1].load_state_dict(self.encoder.adapter_p[2].state_dict())
                self.decoder.adapter_p[1].load_state_dict(self.decoder.adapter_p[2].state_dict())
            else:
                assert task == 'none'

    def get_rd_info(self, fa_idx, weight_index=0, use_large_weight=False):
        if weight_index == 0:
            dist_weights = [1.0, 2.0, 1.5]
        elif weight_index == 1:
            dist_weights = [0.5, 1.2, 0.9]
        else:
            assert False
        dist_weight = dist_weights[fa_idx]
        if use_large_weight:
            dist_weight = 2.0
        return dist_weight

    def forward(self, x, dpb, qp,group_cascade=False):
        index_map = [0, 1, 0, 2, 0, 2, 0, 2]
        rate_gop_size = 4
        assert x.dim() == 5
        B, frame_nums, C, H, W = x.shape
        if frame_nums == 1:
            x = rearrange(x, 'b 1 c h w -> b c h w')
            if dpb["ref_frame"] is None:
                dpb = self.dpb
            else:
                self.frame_idx = 1
            fa_idx = index_map[self.frame_idx]
            result = self.forward_one_frame(x, dpb, qp, fa_idx)
            rd = self.get_rd_info(fa_idx)
            result["rd"] = rd
            self.frame_idx = (self.frame_idx + 1) % rate_gop_size
            self.dpb["ref_frame"] = result["dpb"]["ref_frame"].detach()
            if self.config.model.hyperprior.get('use_hyperprior', False):
                self.dpb["ref_feature"] = result["dpb"]["ref_feature"].detach()
            if self.config.model.memory.get('use_memory', False):
                self.dpb["long_term_mem"] = result["dpb"]["long_term_mem"].detach()
            return result

        bpp_losses = []
        x_hats = []
        rds=[]
        if group_cascade and dpb["ref_frame"] is None:
            dpb = self.dpb
        else:
            assert dpb["ref_frame"] is not None
            dpb["ref_feature"] = None
            self.frame_idx = 1
        dist_weight_index = 0
        if frame_nums > 7:
            dist_weight_index = 1
        if frame_nums >= 15:
            rate_gop_size = 8
        use_large_weight = True
        x_hat_0 = dpb["ref_frame"] # (B, C, H, W)
        for frame_index in range(frame_nums):
            cur_frame = x[:, frame_index, :, :, :] # (B, C, H, W)
            fa_idx = index_map[self.frame_idx]
            result = self.forward_one_frame(cur_frame, dpb, qp, fa_idx)
            rd = self.get_rd_info(fa_idx,weight_index=dist_weight_index,
                        use_large_weight=use_large_weight)
            use_large_weight = False
            self.frame_idx = (self.frame_idx + 1) % rate_gop_size
            bpp_losses.append(result["bpp"])
            x_hats.append(result["x_hat"])
            dpb = result["dpb"]
            rds.append(rd) # list of float
        if group_cascade:
            self.dpb["ref_frame"] = result["dpb"]["ref_frame"].detach()
            self.dpb["ref_feature"] = result["dpb"]["ref_feature"].detach()
            if self.config.model.memory.get('use_memory', False):
                self.dpb["long_term_mem"] = result["dpb"]["long_term_mem"].detach()
        dict4gan = {
                "bpp": torch.stack(bpp_losses, dim=1),
                "x_hat": torch.stack(x_hats, dim=1),
                "rd": torch.tensor(rds, device=x.device).unsqueeze(0),  # (1, frame_nums)
            }
        dict4gan["prev_x_hat"] =  torch.cat((x_hat_0.view(B, 1, 3, H, W), dict4gan["x_hat"][:,:-1,:,:,:]), dim=1)
        result.update(dict4gan)
        return result
