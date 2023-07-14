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

"""Transformer."""

import math

import torch
import torch.nn.init as init  # 导入 PyTorch 的初始化函数模块
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm  # 从 apex 库中导入 FusedLayerNorm 类，并将其命名为 LayerNorm

from .initialize import get_model_parallel_world_size  # 导入当前包的 initialize 模块中的 get_model_parallel_world_size 函数
from .layers import ColumnParallelLinear  # 导入当前包的 layers 模块中的 ColumnParallelLinear 类
from .layers import RowParallelLinear  # 导入当前包的 layers 模块中的 RowParallelLinear 类
from .mappings import gather_from_model_parallel_region  # 导入当前包的 mappings 模块中的 gather_from_model_parallel_region 函数

import deepspeed  # 导入 deepspeed 库

from .random import checkpoint  # 导入当前包的 random 模块中的 checkpoint 函数
from .random import get_cuda_rng_tracker  # 导入当前包的 random 模块中的 get_cuda_rng_tracker 函数

from .utils import divide  # 导入当前包的 utils 模块中的 divide 函数
from .utils import split_tensor_along_last_dim  # 导入当前包的 utils 模块中的 split_tensor_along_last_dim 函数

class PositionalEmbedding(torch.nn.Module):  # 定义 PositionalEmbedding 类，继承自 PyTorch 的 Module 类
    def __init__(self, hidden_size):  # 定义类的初始化函数，接受一个参数 hidden_size
        super(PositionalEmbedding, self).__init__()  # 调用父类的初始化函数

        self.hidden_size = hidden_size  # 将输入的 hidden_size 赋值给类的成员变量
        
        #创建一个变量inv_freq，其作用是生成一个序列，其中包含0到hidden_size间隔为2的数字。然后，这些数字被除以hidden_size并作为10000的幂来计算，最后取倒数。
        #这种计算方法主要用于后面的位置嵌入计算，使得不同位置的嵌入在不同的频率有不同的“波峰”和“波谷”。
        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))  # 计算位置嵌入的倒数频率
        self.register_buffer('inv_freq', inv_freq)  # 注册一个持久缓冲区，将 inv_freq 作为模型的一部分保存下来

    def forward(self, pos_seq, bsz=None):  # 定义类的 forward 函数，接受两个参数 pos_seq 和 bsz
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)  # 计算张量的外积，生成一个正弦输入
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # 通过将正弦和余弦值拼接起来，生成位置嵌入

        if bsz is not None:  # 如果 bsz 不为空
            return pos_emb[None, :, :].expand(bsz, -1, -1)  # 扩展 pos_emb 的大小以匹配 bsz
        else:  # 如果 bsz 为空
            return pos_emb[None, :, :]  # 直接返回 pos_emb



class ParallelCrossAttention(torch.nn.Module):  # 定义一个名为ParallelCrossAttention的类，这是一个PyTorch模型。
    """Parallel cross-attention layer for Transformer"""  # 类的作用说明，表示这是一个为Transformer设计的并行交叉注意力层。

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 output_layer_init_method=None):  # 类的初始化函数，定义一些必要的参数。
        super(ParallelCrossAttention, self).__init__()  # 调用父类（torch.nn.Module）的初始化函数。
        if output_layer_init_method is None:  # 检查是否提供了输出层的初始化方法。
            output_layer_init_method = init_method  # 如果没有提供，则使用普通的初始化方法。
        world_size = get_model_parallel_world_size()  # 获取并行模型的世界大小。
        self.hidden_size_per_partition = divide(hidden_size, world_size)  # 计算每个分区的隐藏层大小。
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)  # 计算每个注意力头的隐藏层大小。
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)  # 计算每个分区的注意力头数量。
        # 创建一个列并行的线性层，用于生成query。
        self.query = ColumnParallelLinear(hidden_size, hidden_size,
                                          gather_output=False,
                                          init_method=init_method)
        # 创建一个列并行的线性层，用于生成key和value。
        self.key_value = ColumnParallelLinear(hidden_size, 2 * hidden_size,
                                              stride=2,
                                              gather_output=False,
                                              init_method=init_method)
        # 创建一个dropout层，用于处理注意力分数。
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # 创建一个行并行的线性层，用于处理输出。
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        # 创建一个dropout层，用于处理输出。
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        # 检查是否配置了deepspeed的检查点。
        if deepspeed.checkpointing.is_configured():
            # 如果配置了检查点，获取CUDA的随机数生成器跟踪器。
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            # 如果配置了检查点，获取deepspeed的检查点函数。
            checkpoint = deepspeed.checkpointing.checkpoint # 获取deepspeed的检查点函数，用于在训练中保存和恢复模型状态。



    
    
    def _transpose_for_scores(self, tensor):  # 私有函数，对输入的3D tensor进行转置操作。
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)  # 计算新的tensor形状。
        tensor = tensor.view(*new_tensor_shape)  # 将原始tensor调整为新的形状。
        return tensor.permute(0, 2, 1, 3)  # 执行一个维度置换操作，并返回结果。
    
    def forward(self, hidden_states, encoder_states, cross_mask):  # 模型的前向传播函数，输入是隐藏状态、编码器状态和交叉掩码。
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        mixed_query_layer = self.query(hidden_states)  # 生成查询层。
        mixed_x_layer = self.key_value(encoder_states)  # 生成键值层。
        (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)  # 将键值层拆分为键层和值层。
    
        query_layer = self._transpose_for_scores(mixed_query_layer)  # 将查询层转置。
        key_layer = self._transpose_for_scores(mixed_key_layer)  # 将键层转置。
        value_layer = self._transpose_for_scores(mixed_value_layer)  # 将值层转置。
    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 计算注意力分数。
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)  # 对注意力分数进行缩放。
    
        if cross_mask is not None:  # 检查是否提供了交叉掩码。
            attention_scores = torch.mul(attention_scores, cross_mask) - 10000.0 * (1.0 - cross_mask)  # 如果提供了交叉掩码，将其应用到注意力分数上。
    
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)  # 计算注意力概率。
    
        with get_cuda_rng_tracker().fork():  # 启动一个新的CUDA随机数生成器跟踪器上下文。
            attention_probs = self.attention_dropout(attention_probs)  # 对注意力概率进行dropout操作。
    
        context_layer = torch.matmul(attention_probs, value_layer)  # 计算上下文层。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 对上下文层进行置换操作并确保它在内存中是连续的。
    
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)  # 计算新的上下文层形状。
        context_layer = context_layer.view(*new_context_layer_shape)  # 将上下文层调整为新的形状。
    
        output = self.dense(context_layer)  # 计算输出。
        output = self.output_dropout(output)  # 对输出进行dropout操作。
    
        return output  # 返回输出。



class ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT2.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None, relative_encoding=False,
                 performer=False, attention_scale=1.0):
        super(ParallelSelfAttention, self).__init__()
        self.performer = performer
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        self.relative_encoding = relative_encoding
        self.attention_scale = attention_scale
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        if relative_encoding:
            self.relative = ColumnParallelLinear(hidden_size, hidden_size, gather_output=False,
                                                 init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def _rel_shift(x, zero_triu=False):
        # ql x kl x bsz x h
        # bsz x h x ql x kl
        zero_pad = torch.zeros((*x.size()[:-2], x.size(-2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:-2], x.size(-1) + 1, x.size(-2))

        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)

        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        if self.relative_encoding:
            relative_layer = self.relative(position_embeddings)
            relative_layer = self._transpose_for_scores(relative_layer)  # 1 (bsz) x n_head x klen x d_head
            # Raw attention scores. [b, np, qs, ks]
            rw_head_q = query_layer + r_w_bias.unsqueeze(1)
            ac_score = torch.matmul(rw_head_q, key_layer.transpose(-1, -2))
            rr_head_q = query_layer + r_r_bias.unsqueeze(1)
            bd_score = torch.matmul(rr_head_q, relative_layer.transpose(-1, -2))
            bd_score = self._rel_shift(bd_score)  # qlen x klen x bsz x n_head
            # bd_score = bd_score.permute(2, 3, 0, 1) # bsz n_head qlen klen

            attention_scores = ac_score + bd_score
            attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        else:
            if self.attention_scale > 1.0:
                # Raw attention scores. [b, np, s, s]
                attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_scale),
                                            key_layer.transpose(-1, -2) / math.sqrt(
                                                self.hidden_size_per_attention_head * self.attention_scale))
            else:
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2) / math.sqrt(
                    self.hidden_size_per_attention_head))

        # Apply the left to right attention mask.
        attention_scores = torch.mul(attention_scores, ltor_mask)
        if self.attention_scale > 1.0:
            max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
            attention_scores -= max_attention_scores
            attention_scores *= self.attention_scale
        # if torch.distributed.get_rank() == 0:
        #     print(min_attention_scores, attention_scores.max().item())
        attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
        # Attention probabilities. [b, np, s, s]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + \
                                  (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class ParallelMLP(torch.nn.Module):
    """MLP for GPT2.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class ParallelDecoderLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None):
        super(ParallelDecoderLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.self_attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

        # Layernorm after the self attention.
        self.post_self_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.cross_attention = ParallelCrossAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method
        )

        # Layernorm after the cross attention.
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, encoder_states, ltor_mask, cross_mask=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        self_attention_output = self.self_attention(layernorm_output, ltor_mask)
        # Residual connection.
        self_layernorm_input = hidden_states + self_attention_output
        # Layer norm post the self attention.
        self_layernorm_output = self.post_self_layernorm(self_layernorm_input)
        # Cross attention
        attention_output = self.cross_attention(self_layernorm_output, encoder_states, cross_mask)
        # Residual connection
        layernorm_input = self_layernorm_input + attention_output
        # Layer norm post the cross attention
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output
        return output


class ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT2.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 relative_encoding=False,
                 performer=False,
                 attention_scale=1.0):
        super(ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            relative_encoding=relative_encoding,
            performer=performer,
            attention_scale=attention_scale)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, position_embeddings=None, r_w_bias=None, r_r_bias=None, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT2ParallelTransformer(torch.nn.Module): # 定义一个名为 GPT2ParallelTransformer 的类，继承自 PyTorch 的 nn.Module
    """GPT-2 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """
    '''这是类的初始化函数，用于初始化该类的一些参数和属性。它接收多个参数，包括：层数、隐藏层大小、注意力头的数量、最大序列长度、最大记忆长度、
    嵌入的dropout概率、注意力的dropout概率、输出的dropout概率、是否进行激活检查点、进行检查点的层数、层归一化的epsilon、
    初始化方法的标准差、是否对输出权重进行缩放初始化、是否使用相对编码、是否使用块位置编码、是否使用执行者、是否使用解码器层和注意力的比例。'''
    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 max_memory_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 performer=False,
                 use_decoder_layer=False,
                 attention_scale=1.0,
                 ):  
        # 这是类的初始化函数，用于初始化该类的一些参数和属性
                 
        super(GPT2ParallelTransformer, self).__init__()  # 对父类 `torch.nn.Module` 的初始化
        
        self.hidden_size = hidden_size  # 设置隐藏层大小的属性
        self.checkpoint_activations = checkpoint_activations  # 设置是否进行激活检查点的属性
        self.checkpoint_num_layers = checkpoint_num_layers  # 设置进行检查点的层数的属性
        self.max_memory_length = max_memory_length  # 设置最大记忆长度的属性
        self.performer = performer  # 设置是否使用执行者的属性
        self.use_decoder_layer = use_decoder_layer  # 设置是否使用解码器层的属性
        
        assert not (performer and relative_encoding)  # 断言如果 `performer` 和 `relative_encoding` 都为 True，则引发错误

        output_layer_init_method = None  # 初始化输出层初始化方法为None
        if use_scaled_init_for_output_weights:  # 如果使用缩放的初始化方法进行权重初始化
            output_layer_init_method = scaled_init_method(init_method_std, num_layers)  # 则设置output_layer_init_method为scaled_init_method的输出
        
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)  # 创建一个dropout层，丢弃概率为embedding_dropout_prob
        
        self.relative_encoding = relative_encoding  # 设置relative_encoding属性
        self.block_position_encoding = block_position_encoding  # 设置block_position_encoding属性
        
        if relative_encoding:  # 如果启用了relative_encoding
            self.position_embeddings = PositionalEmbedding(hidden_size)  # 创建一个位置嵌入层
        
            world_size = get_model_parallel_world_size()  # 获取模型并行的世界大小
        
            # 计算每个注意力头的隐藏层大小和每个分区的注意力头数
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
            self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        
            # 创建两个bias参数，并进行初始化
            self.r_w_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
            self.r_r_bias = torch.nn.Parameter(
                torch.Tensor(self.num_attention_heads_per_partition, self.hidden_size_per_attention_head))
                
            self.r_w_bias.model_parallel = True  # 将r_w_bias的model_parallel属性设置为True
            self.r_r_bias.model_parallel = True  # 将r_r_bias的model_parallel属性设置为True
            
            # 在不计算梯度的情况下，将r_w_bias和r_r_bias初始化为零
            with torch.no_grad():
                self.r_w_bias.zero_()
                self.r_r_bias.zero_()
        
        else:  # 如果relative_encoding为False
            # 如果block_position_encoding为True，那么对max_sequence_length加1，否则不变
            if block_position_encoding:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
                torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)  # 对block_position_embeddings的权重进行正态分布初始化
            else:
                self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)  # 创建一个位置嵌入层
                
            torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)  # 对position_embeddings的权重进行正态分布初始化
                

        def get_layer():  # 定义一个内部函数，用于获取不同类型的层
            if use_decoder_layer:  # 如果使用解码器层
                return ParallelDecoderLayer(  # 返回ParallelDecoderLayer实例
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    unscaled_init_method(init_method_std),
                    output_layer_init_method=output_layer_init_method
                )
            else:  # 否则
                return ParallelTransformerLayer(  # 返回ParallelTransformerLayer实例
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    unscaled_init_method(init_method_std),
                    output_layer_init_method=output_layer_init_method,
                    relative_encoding=relative_encoding,
                    performer=performer,
                    attention_scale=attention_scale)
        
        self.layers = torch.nn.ModuleList([get_layer() for _ in range(num_layers)])  # 创建包含num_layers个get_layer返回的层的模块列表
        
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)  # 创建最后的层规范化 (Layer Normalization) 层
        
        if deepspeed.checkpointing.is_configured():  # 如果配置了deepspeed的检查点功能
            global get_cuda_rng_tracker, checkpoint  # 声明两个全局变量
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker  # 获取deepspeed的CUDA随机数生成器追踪器
            checkpoint = deepspeed.checkpointing.checkpoint  # 获取deepspeed的检查点函数
            #这段代码主要定义了获取不同类型的层的函数 get_layer，并用此函数生成了一系列的模型层。这些层存储在 self.layers 中。此外，这段代码还创建了一个最后的层规范化 (Layer Normalization) 层，以及设置了一些deepspeed的检查点功能相关的变量。

    def forward(self, hidden_states, position_ids, attention_mask, memory_states=None, encoder_states=None,
                return_memory=False, detach_memory=True):
        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0
        key_length = query_length + memory_length
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if self.performer:
            assert is_scalar, 'attention_mask should be a scalar to indicate the seperation position.'
            assert memory_length == 0, 'Do not support transformer-xl.'
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, :sep] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m

            if not self.performer:
                attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
        else:
            attention_mask = attention_mask.type_as(hidden_states)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length:]

        if self.relative_encoding:
            position_sequence = torch.arange(key_length - 1, -1, -1.0, device=hidden_states.device,
                                             dtype=hidden_states.dtype)
            position_embeddings = self.position_embeddings(position_sequence)
            # Apply dropout
            position_embeddings = self.embedding_dropout(position_embeddings)
        else:
            if self.block_position_encoding:
                position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
            if self.block_position_encoding:
                block_position_embeddings = self.block_position_embeddings(block_position_ids)
                hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            if detach_memory:
                return _hidden_states.detach()
            return _hidden_states

        if self.max_memory_length > 0 or return_memory:
            mem_layers = [check_detach(hidden_states)]
        else:
            mem_layers = []

        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_, inputs = inputs[0], inputs[1:]
                if self.relative_encoding:
                    inputs, mems_ = inputs[:4], inputs[4:]
                else:
                    inputs, mems_ = inputs[:1], inputs[1:]
                for i, layer in enumerate(layers_):
                    mem_i_ = mems_[i] if mems_ else None
                    x_ = layer(x_, *inputs, mem=mem_i_)
                    if self.max_memory_length > 0 or return_memory:
                        mem_layers.append(check_detach(x_))
                return x_

            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                if memory_states:
                    args += memory_states[l: l + chunk_length]
                hidden_states = checkpoint(custom(l, l + chunk_length), *args)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask] if not self.use_decoder_layer else [hidden_states,
                                                                                           encoder_states,
                                                                                           attention_mask]
                if self.relative_encoding:
                    args += [position_embeddings, self.r_w_bias, self.r_r_bias]
                mem_i = memory_states[i] if memory_states else None
                hidden_states = layer(*args, mem=mem_i)
                if self.max_memory_length > 0 or return_memory:
                    mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        if self.max_memory_length > 0 or return_memory:
            mem_layers = self.update_mems(mem_layers, memory_states, return_memory=return_memory)

        return (output, mem_layers)

    def update_mems(self, hiddens, mems, return_memory=False):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length
        if not return_memory:
            new_memory_length = min(self.max_memory_length, new_memory_length)
        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length:], hiddens[i]), dim=1))
        return new_mems
