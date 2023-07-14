# GLM-Explained
GLM-Explained
这个项目主要是LLM原理。若是想弄大模型的应用，那么来错地方，建议看[ChatGLM2-Explained](https://github.com/ArtificialZeng/ChatGLM2-6B-Explained)

此外，大模型还基于两个非常重要的基础库，那便是[transformers](https://github.com/ArtificialZeng/tranformers-expalined)，和[pytorch](https://github.com/ArtificialZeng/pytorch-explained)，同样这两个库也有关键代码的逐行解析版本。

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
   



