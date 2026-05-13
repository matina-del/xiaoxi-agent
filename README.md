# 小夕 · 并夕夕智能客服 Agent

> 基于 Python + OpenAI SDK 构建的电商客服 Agent，覆盖 System Prompt 设计、结构化输出、上下文压缩、ReAct 范式、MCP 工具协议，以及即将落地的 RAG 检索增强。

---

## 目录

- [1. System Prompt 设计](#1-system-prompt-设计)
- [2. Structured Output](#2-structured-output)
- [3. 上下文压缩策略](#3-上下文压缩策略)
- [4. ReAct Agent 范式](#4-react-agent-范式)
- [5. MCP 工具协议](#5-mcp-工具协议)
- [6. 下一步：RAG 检索增强](#6-下一步rag-检索增强)

---

## 1. System Prompt 设计

System Prompt 是整个客服 Agent 的行为基座，从四个维度约束模型：

**角色设定**
明确「你是谁」和「你的语气风格」——小夕是并夕夕平台的专属客服，语气亲切、简洁、有温度，不使用官腔和推卸式表达。

**能力边界**
显式声明能做什么、不能做什么。能做：查订单、查物流、受理退换货申请、解答平台政策。不能做：承诺超出权限的补偿、修改已核销订单、代替用户操作账户。

**回复规范**
先安抚情绪，再解决问题；主动给出选项，不让用户猜下一步。例如：用户投诉包裹破损，先表达歉意，再给「申请退款 / 补发商品 / 联系商家」三个选项，而不是直接问「你想怎么处理」。

**安全约束**
不编造订单信息、物流状态、退款进度；不越权承诺（如「我保证明天到货」）；遇到无法处理的情况，转人工而不是强行回答。

---

## 2. Structured Output

客服系统的消费方不只是「看回复内容的人」，后端逻辑同样需要知道：

- 用户意图是什么（查订单？退货？投诉？咨询？）
- 置信度多高
- 是否需要转人工

使用 OpenAI SDK 的 `response_format` 配合 **Pydantic BaseModel**，让模型直接输出严格 JSON，无需手写正则解析，返回即为 Python 对象，直接 `.intent`、`.reply`、`.confidence` 点出来用。

```python
class CustomerServiceResponse(BaseModel):
    intent: str          # "query_order" | "return" | "complaint" | "inquiry"
    confidence: float    # 0.0 ~ 1.0
    reply: str           # 面向用户的回复内容
    transfer_human: bool # 是否转人工

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=CustomerServiceResponse,
)

result = response.choices[0].message.parsed
print(result.intent)         # "return"
print(result.transfer_human) # False
```

结构化输出让客服 Agent 从「聊天机器人」升级为「可被系统集成的服务组件」。

---

## 3. 上下文压缩策略

### 问题

对于长会话 Agent，如果每轮都把完整 history 塞回给 LLM：

- Token 消耗随轮数线性增长，成本快速上升
- 模型注意力被稀释在巨长的历史中，越聊越「健忘」，Agent 效果随轮数下降

### 策略

**保留最近 N 轮原文，更早的交给 LLM 自己压缩成摘要。**

触发条件：消息超过 10 条时触发一次压缩，保留最近 3 条原文，其余压缩为一段结构化摘要。

> Claude Code 的参考阈值：达到最大上下文窗口的约 70% 时自动触发压缩。

### 关键：摘要不是流水账

摘要不能写成「用户之前说了什么、客服回复了什么」的流水记录，而是提取**记忆骨架**——不管聊多久，这五条信息不能丢：

| 字段 | 说明 |
|------|------|
| 用户身份 | 昵称、会员等级、历史投诉记录等 |
| 订单号 | 商品 | 本次会话涉及的核心对象 |
| 核心诉求 | 用户最终想要什么结果 |
| 客服已承诺的事 | 防止前后矛盾、重复承诺 |
| 未回答的问题 | 避免用户重复追问被忽略的点 |

这五条是针对电商客服场景设计的「记忆骨架」。换个垂类（医疗、法律、教育），骨架字段会变，但「提取结构化核心而非堆砌流水账」的设计思路是通用的。

---

## 4. ReAct Agent 范式

**ReAct = Reasoning + Acting**，大模型领域最经典的 Agent 范式。

### 执行流程

以「帮我查一下订单 ORD-001 的物流到哪了」为例：

```
💭 思考：用户要查物流，我需要调用物流查询工具
🔧 调用：query_logistics(order_id="ORD-001")
📋 结果：顺丰 SF123456，1月18日到达上海浦东，正在派送中
💭 思考：拿到数据了，可以回复用户了
🤖 回复：您的快递目前正在上海浦东派送中，预计今日送达……
```

模型在每一步都先推理「我需要做什么」，再决定「调用哪个工具」，拿到结果后再推理「是否够用、是否需要继续」，直到可以给出最终回复。

ReAct 让 Agent 的决策过程可观测、可调试，而不是一个不透明的「输入→输出」黑盒。

---

## 5. MCP 工具协议

### 问题

查订单、查物流这些工具写成本地函数可以跑通，但真实系统中，**订单服务、物流服务、退款服务是不同团队维护的不同微服务**，不可能把所有工具函数都塞进 Agent 代码。

### MCP 是什么

**MCP（Model Context Protocol）** 是工具调用的标准协议，可以理解成「Agent 世界的 USB 接口」。

- 工具方按协议暴露接口
- Agent 启动时自动发现有哪些工具、参数是什么
- 不需要在 Agent 侧提前写死工具 schema

### 接入方式

```
┌─────────────────────────────────────┐
│            MCP Server               │
│  (独立进程，Streamable HTTP 监听)    │
│                                     │
│  - query_order()                    │
│  - query_logistics()                │
│  - apply_refund()                   │
│  - query_product()                  │
└──────────────┬──────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────┐
│           ToolManager               │
│  启动时连接 MCP Server，自动发现工具  │
│  MCP schema → OpenAI function 格式  │
│  工具调用请求转发至 MCP Server        │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│          ReAct Agent Loop           │
│  工具列表和调用入口统一走 ToolManager │
│  核心循环本身只改了两行              │
└─────────────────────────────────────┘
```

### 为什么选 Streamable HTTP 而非 stdio

`stdio` 最简单，但工具作为子进程运行，真实场景中工具是独立部署的。Streamable HTTP 更贴近实际生产部署，且天然支持多个 Agent 实例同时连接同一个 MCP Server。

---

## 6. 下一步：RAG 检索增强

当前 Agent 的知识完全依赖工具返回的结构化数据。下一步引入 **RAG（Retrieval-Augmented Generation）**，让 Agent 能检索：

- 📦 商品库（规格、库存、价格）
- ❓ FAQ（高频问题标准答案）
- 📋 退换货政策（平台规则全文）

Agent 不再只靠工具数据回答问题，而是能结合检索到的知识片段，给出更准确、更有据可查的回复。

---

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| LLM SDK | OpenAI SDK |
| 结构化输出 | Pydantic v2 |
| Agent 范式 | ReAct |
| 工具协议 | MCP (Streamable HTTP) |
| 下一步 | RAG（向量检索） |
