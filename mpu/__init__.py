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

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy  # 从cross_entropy模块中导入vocab_parallel_cross_entropy，这是一个并行化的交叉熵损失函数，特别适合于处理词汇并行的场景。

from .data import broadcast_data  # 从data模块中导入broadcast_data函数。这个函数能广播数据到多个设备，以在并行计算环境中进行数据的同步。

from .grads import clip_grad_norm  # 从grads模块中导入clip_grad_norm函数。这个函数可以对模型的梯度进行裁剪，以防止梯度爆炸。

from .initialize import destroy_model_parallel  # 从initialize模块中导入destroy_model_parallel函数，用于销毁并行模型。
from .initialize import get_data_parallel_group  # 从initialize模块中导入get_data_parallel_group函数，用于获取数据并行的组。
from .initialize import get_data_parallel_rank  # 从initialize模块中导入get_data_parallel_rank函数，用于获取数据并行的等级。
from .initialize import get_data_parallel_world_size  # 从initialize模块中导入get_data_parallel_world_size函数，用于获取数据并行的世界大小。
from .initialize import get_model_parallel_group  # 从initialize模块中导入get_model_parallel_group函数，用于获取模型并行的组。
from .initialize import get_model_parallel_rank  # 从initialize模块中导入get_model_parallel_rank函数，用于获取模型并行的等级。
from .initialize import get_model_parallel_src_rank  # 从initialize模块中导入get_model_parallel_src_rank函数，用于获取模型并行的源等级。
from .initialize import get_model_parallel_world_size  # 从initialize模块中导入get_model_parallel_world_size函数，用于获取模型并行的世界大小。
from .initialize import initialize_model_parallel  # 从initialize模块中导入initialize_model_parallel函数，用于初始化并行模型。
from .initialize import model_parallel_is_initialized  # 从initialize模块中导入model_parallel_is_initialized函数，用于检查模型并行是否已经初始化。

from .layers import ColumnParallelLinear  # 从layers模块中导入ColumnParallelLinear类，这是一个列并行的线性层。
from .layers import ParallelEmbedding  # 从layers模块中导入ParallelEmbedding类，这是一个并行的嵌入层。
from .layers import RowParallelLinear  # 从layers模块中导入RowParallelLinear类，这是一个行并行的线性层。
from .layers import VocabParallelEmbedding  # 从layers模块中导入VocabParallelEmbedding类，这是一个词汇并行的嵌入层。

from .mappings import copy_to_model_parallel_region  # 从mappings模块中导入copy_to_model_parallel_region函数，用于复制数据到模型并行的区域。
from .mappings import gather_from_model_parallel_region  # 从mappings模块中导入gather_from_model_parallel_region函数，用于从模型并行的区域收集数据。
from .mappings import reduce_from_model_parallel_region  # 从mappings模块中导入reduce_from_model_parallel_region函数，用于减少模型并行区域的数据。
from .mappings import scatter_to_model_parallel_region  # 从mappings模块中导入scatter_to_model_parallel_region函数，用于分散数据到模型并行的区域。

from .random import checkpoint  # 从random模块中导入checkpoint函数，用于并行计算环境中的checkpoint处理。
from .random import partition_activations_in_checkpoint  # 从random模块中导入partition_activations_in_checkpoint函数，用于处理激活函数在checkpoint中的分区。
from .random import get_cuda_rng_tracker  # 从random模块中导入get_cuda_rng_tracker函数，用于获取CUDA的随机数生成器追踪器。
from .random import model_parallel_cuda_manual_seed  # 从random模块中导入model_parallel_cuda_manual_seed函数，用于设置并行CUDA的随机种子。

from .transformer import GPT2ParallelTransformer  # 从transformer模块中导入GPT2ParallelTransformer类，这是GPT-2模型的并行化版本。
from .transformer import LayerNorm  # 从transformer模块中导入LayerNorm类，这是一种层归一化操作。

'''

在Python中，__init__.py文件是一种特殊的模块文件，它被放在一个包的目录中，以使这个目录被Python认为是一个包。当你导入一个包时，__init__.py文件被自动执行。在早期的Python版本中，如果一个文件夹内没有__init__.py文件，Python就不会把该文件夹视为是一个包。

__init__.py文件的主要作用有：

初始化Python包，有时你可能需要在包被导入时执行一些初始化操作。
确定包的公开接口，也就是确定哪些名字会被导出到外部使用。你可以通过定义__all__变量，或者在__init__.py中直接导入一些模块或变量等，这样在外部就可以通过包名直接使用这些模块或变量。
在以上提供的__init__.py文件中，它主要的作用是从模型并行处理工具(mpu)的其他各个子模块中导入了一些类和函数，这些类和函数定义了并行计算环境中模型和数据的处理方式。它们被导入到__init__.py文件中，这样在其他地方使用mpu包时，就可以直接通过from mpu import ...的方式来使用这些类和函数，而不需要写出它们完整的模块路径，这样可以大大简化代码的编写。
'''

