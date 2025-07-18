
# 🤖 从零构建大语言模型：个人学习与项目复现实录

本项目是基于《从零构建大模型》（Sebastian Raschka 著）一书进行的系统性复现实践，  
旨在通过手动搭建大语言模型的关键模块，深入理解其底层结构与原理，  
并逐步构建个人的 AI 应用工程师技能体系与实战项目集。



---

## 📚 项目模块目录

| 模块章节 | 内容简介 | 状态 |
|----------|----------|------|
| 第2章 | 分词与嵌入：Tokenizer、Embedding 原理与复现 | ✅ 已完成 |
| 第3章 | 注意力机制：Self-Attention 实现与动手实践 | ✅ 已完成 |
| 第4章 | 编码器结构：Transformer 编码堆叠搭建 |  ✅ 已完成 |
| 第5章 | 自监督训练流程：损失函数、训练管线构建 | ⏳ 待学习 |
| ... | 更多章节待逐步推进 | - |

---

## 🧠 学习目标

- 深度理解 LLM 架构核心模块：Tokenizer、Embedding、Attention、Transformer 等；
- 掌握 PyTorch 从零手写模型模块与训练代码的能力；
- 能独立构建小型语言模型管线并在本地运行、测试；
- 将理论知识转化为结构化笔记、简历项目与作品集内容。

---

## 🗂️ 当前项目结构

```
├── src/                     # 所有核心模块代码
│   ├── ch2_tokenizer_and_embedding/  # 分词器和嵌入模块（第二章）
│   ├── ch3_attention/                # 注意力机制实现（第三章）
│   │   ├── attention_stage2.py       # 多阶段注意力构建
│   │   ├── casual_attention.py       # 因果注意力实现
│   │   ├── multi_head_attention.py   # 多头注意力封装
│   │   ├── simple_attention.py       # 简化单头注意力
│   │   └── ...
│   └── ch4_llm_gpt/                  # GPT 模型完整构建（第四章）
│       ├── gpt_model.py              # GPT模型的完整实现
│       ├── gpt_frame.py              # DummyGPTModel 主体结构
│       ├── layer_gelu.py             # GELU 激活函数封装
│       ├── layer_norm.py             # 自定义 LayerNorm 层
│       ├── resnet.py                 # 残差连接模块
│       ├── transformer_.py           # TransformerBlock 封装
│       └── __init__.py
│
├── notes/                    # 每章学习总结笔记
│   ├── chapter2_summary.md
│   ├── chapter3_summary.md
│   └── chapter4_summary.md 
│
├── the-verdict.txt           # 示例文本
└── README.md                 # 本说明文档



## ✍️ 关于作者

本项目由正在转型中的非科班学习者 [@weixiaocan](https://github.com/weixiaocan) 完成。  
当前正通过“原理学习 + 模块复现 + 应用实践”的路径转向 AI 应用工程师方向，  
欢迎关注进度，也欢迎同行交流与建议。

---

## ⭐ 后续规划

- 项目代码与笔记持续更新中
- 后期加入小项目整合：如微调、推理接口、Agent 应用、个人 Demo
- 打通学习路径 → 实战输出 → 求职展示的完整链条
