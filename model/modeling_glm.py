# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import mpu
from model.prompt import PromptSpell
from utils import print_rank_0


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_



class GLMModel(nn.Module):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0,
                 ):
        # 调用父类的初始化函数
        super(GLMModel, self).__init__()  # 初始化nn.Module的基类

        # 初始化一些基本的参数
        self.parallel_output = parallel_output  # 设置并行输出
        self.output_predict = output_predict  # 设置输出预测
        self.hidden_size = hidden_size  # 隐藏层的大小

        # 初始化权重的方法，使用正态分布，标准差为0.02
        init_method = init_method_normal(std=0.02)

        # 设置词嵌入层，这里使用了并行词嵌入
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)  # 初始化词嵌入层

        # 设置Transformer层
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       relative_encoding=relative_encoding,
                                                       block_position_encoding=block_position_encoding)  # 初始化Transformer层

        # 如果提供了spell_length，那么初始化PromptSpell
        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)  # 初始化PromptSpell层

    # 冻结Transformer的参数
    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"  # 日志字符串
        # 禁止词嵌入层的参数梯度更新
        self.word_embeddings.requires_grad_(False)  
        # 禁止transformer层的参数梯度更新
        self.transformer.requires_grad_(False)  
        # 如果指定了前缀层，则开启前缀层的参数梯度更新
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"  # 更新日志字符串
            for i in range(tune_prefix_layers):
                # 允许前缀层的参数梯度更新
                self.transformer.layers[i].requires_grad_(True)  
        # 打印日志信息
        print_rank_0(log_str)  

    # 前向传播函数
    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        # 获取batch的大小
        batch_size = input_ids.size(0)  
        # 获取词嵌入
        words_embeddings = self.word_embeddings(input_ids)  
        embeddings = words_embeddings
        # 如果有提示位置，处理提示
        if prompt_pos is not None:
            embeddings = embeddings.clone()  # 复制嵌入向量
            prompt_embeds = self.prompt_spell()  # 获取提示嵌入
            # 创建一个与batch大小相同的索引数组
            batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)  
            # 将提示嵌入插入到指定位置
            embeddings[batch_index, prompt_pos] = prompt_embeds  
        # Transformer.
        # 通过Transformer层
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems,
                                              return_memory=return_memory, detach_memory=detach_memory)  
        # 获取Transformer层的输出
        logits, hidden_layers = transformer_output  
        # 设置输出
        outputs = hidden_layers  

        # 如果开启了output_predict
        if self.output_predict:
        # 并行获取logits
        logits_parallel = mpu.copy_to_model_parallel_region(
            logits)  # 将logits复制到并行模型区域
        # 进行线性变换，这里使用了词嵌入的权重作为线性变换的权重
        logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)  

        # 如果开启了并行输出
        if self.parallel_output:
            # 返回并行logits和输出
            return (logits_parallel, *outputs)  

        # 如果没有开启并行输出，那么需要收集并行模型区域的logits，并返回
        return (mpu.gather_from_model_parallel_region(logits_parallel), *outputs)  
    else:
        # 如果没有开启output_predict，直接返回logits和输出
        return (logits, *outputs)  

#这段代码定义了一个名为EncoderDecoder的类，该类继承自torch.nn.Module。这个类实现了一个Seq2Seq的Transformer模型，该模型包括一个编码器（Encoder）和一个解码器（Decoder）。
class EncoderDecoder(nn.Module):
    """Seq2Seq Transformer Model
    The output of the forward method are the logits (parallel or serial depending on the `parallel_output` flag).
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 output_predict=True
                 ):
        # 调用父类初始化函数
        super(EncoderDecoder, self).__init__()  # 初始化nn.Module的基类

        # 初始化一些基本的参数
        self.parallel_output = parallel_output  # 设置并行输出
        self.output_predict = output_predict  # 设置输出预测

        # 初始化权重的方法，使用正态分布，标准差为0.02
        init_method = init_method_normal(std=0.02)

        # 设置词嵌入层，这里使用了并行词嵌入
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)  # 初始化词嵌入层

        # 设置Transformer的编码器
        self.encoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers)  # 初始化编码器

        # 设置Transformer的解码器
        self.decoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers,
                                                   use_decoder_layer=True)  # 初始化解码器，这里使用了解码器层

    def forward(self, source_ids, target_ids, source_position_ids, target_position_ids, source_mask, target_mask):  
    # 定义 Transformer 模型的前向传播函数
    
    # Embeddings.
    source_embeddings = self.word_embeddings(source_ids)  # 对源语言的 token ID 进行词嵌入操作
    target_embeddings = self.word_embeddings(target_ids)  # 对目标语言的 token ID 进行词嵌入操作

    # Transformer.
    encoder_output, _ = self.encoder(source_embeddings, source_position_ids, source_mask)  # 将嵌入后的输入传递给编码器
    decoder_output, _ = self.decoder(target_embeddings, target_position_ids, target_mask)  # 将嵌入后的输入传递给解码器

    if self.output_predict:  # 如果设置了 output_predict 属性
        # Parallel logits.
        output_parallel = mpu.copy_to_model_parallel_region(decoder_output)  # 将解码器的输出复制到模型并行区域
        logits_parallel = F.linear(output_parallel, self.word_embeddings.weight)  # 对复制的输出进行线性变换

        if self.parallel_output:  # 如果设置了 parallel_output 属性
            return (logits_parallel,)  # 直接返回并行区域的 logits

        return (mpu.gather_from_model_parallel_region(logits_parallel),)  # 从模型的并行区域收集 logits 并返回

    else:
        return (decoder_output,)  # 如果没有设置 output_predict 属性，只返回解码器的输出


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
