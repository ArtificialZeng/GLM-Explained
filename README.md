# GLM-Explained
GLM-Explained

* [/model/](./model/)
   * [modeling_glm.py](/model/modeling_glm.py)
     * class EncoderDecoder(nn.Module):
* [/mpu/](./mpu/) 并行处理单元
  * [\__init__.py](/mpu/__init__.py)
  * [transformer.py](/mpu/transformer.py)
    * class PositionalEmbedding(torch.nn.Module):
    * class ParallelCrossAttention(torch.nn.Module):
    * class ParallelSelfAttention(torch.nn.Module):
    * class ParallelMLP(torch.nn.Module):
    * class ParallelDecoderLayer(torch.nn.Module):
    * class ParallelTransformerLayer(torch.nn.Module):
    * class GPT2ParallelTransformer(torch.nn.Module):
   



