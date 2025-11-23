# Forwarder Enhancement Plan

目标：让当前 Python SDK 驱动的转发器（已弃用 Cherry/Node 桥）在“上游 Responses → 下游 ChatCompletions”这一固定方向上，支持更多 Responses 能力，同时给下游继续提供 Chat Completions 兼容体验（不再考虑两端互换）。核心阶段：

1. ✅ **下游流式转发**（已上线：Responses → Chat SSE）
2. ✅ **Reasoning/Thinking 内容**（已上线：`[thinking]...[/thinking]` + metadata + auto `include` 补全）
3. ✅ **工具/函数调用等高级字段**（已上线：工具调用/响应全面互通）
4. 🚧 **多模态输入输出**
5. 🚧 **观测性 / 自动化验证**

以下是详细方案：

---

## 1. 下游流式转发

### 现状
- Flask 端 `/v1/chat/completions` 默认 `stream: true`，使用 Python SDK 的 `client.responses.stream()` 实时消费 SSE，并逐条转换成 Chat Completions chunk（含 reasoning delta）。
- 非 streaming 场景仍然 `stream.until_done()` 后一次性返回。

### 后续
- 监控日志量（大量 delta 会刷屏）：可增加简单的节流/采样或为 `response.output_text.delta` 关闭日志。

---

## 2. Reasoning / Thinking 内容

### 现状
- Responses 返回的 reasoning block 会被 `translate_respond_to_chat()` 收集并注入 `[thinking]...[/thinking]`，同时写入 `choice.message.metadata["reasoning"]`。
- 流式模式会在 `response.reasoning_text.delta/done` 事件间插入 `[thinking]` 块，并将最终 reasoning 文本附带在 finish chunk metadata。
- 当调用方设置 `reasoning`（或 `reasoning_effort`）时，forwarder 会自动补上 `include=["reasoning"]`，避免因为遗漏 include 而拿不到 reasoning 块。

### 后续
- 根据消费方反馈，决定是否改用 `content` 数组（OpenAI chat 格式支持多段）而不是内联 `[thinking]`。
- 若需要 reasoning summary（`reasoning.summary`），可在 metadata 中额外暴露。

---

## 3. Tools / Function Calls

### 现状
- Chat → Responses：已经能够解析 `tool_calls`/`function_call`/`role:"tool"` 消息，并将其转换成 Responses API 的 `function_call` / `function_call_output` 输入块，同时将 `tools`、`tool_choice`、旧版 `functions` 全量映射。
- Responses → Chat：会遍历 `output` 中的工具调用项，将其回填到 ChatCompletions 的 `tool_calls` 和 `function_call` 字段，必要时把 `finish_reason` 设为 `tool_calls`，流式情况下也能实时输出相应 delta。
- Payload 侧还补齐了 `tool_choice` 兼容逻辑（兼容旧字段 `function_call`），并保持 reasoning、其他超参的透传。

### 后续
- 验证并补齐 Responses 侧除 `function_call` 以外的内建工具（如 `file_search`、`code_interpreter`、MCP）映射，避免未来多模态阶段重复实现。
- 加一套端到端回归用例（或手动脚本）覆盖“调用工具 → tool 响应 → 模型继续输出”完整链路，确保 streaming 与非 streaming 行为一致。

---

## 4. 图片 / 文档输入

### 现状
- Chat → Responses：我们当前 `convert_chat_messages_to_respond_input` 只处理字符串文本；如果 ChatCompletion `message.content` 包含图片（`{"type":"image_url",...}`）或上游的 Responses 需要 `input_file`，还未处理。
- Responses → Chat：同样忽略了 `input_image`、`input_file` 等内容，也没有把 Responses 的文档/图片输出还原成 ChatCompletion 的多模态结构。

### 需求
- 项目 B（只能发 ChatCompletions）在消息里嵌入图片/文件时，forwarder 应将这些内容转成 Responses API 支持的 `input_image` / `input_file` 格式；项目 A 返回图片或文档输出时也能映射回 ChatCompletion 的 `image_url` 或 `attachments`。

### 主要改动
1. **Chat → Responses**
   - 遍历 `message["content"]`，区分不同内容块：
     - 文本 → `input_text`
     - `{"type":"image_url","image_url":{"url":...}}` → Responses 的 `input_image`（需要 base64 还是 URL 取决于后端接受方式）
     - `role: "tool"` + `content` → `tool_response`
   - 对于附件（如 PDF），需要额外上传/引用方式：可能参考 Cherry Studio 的 `OpenAIResponseAPIClient` 中 `convertMessageToSdkParam` 的 `input_file` 构造逻辑，支持 base64 files。
2. **Responses → Chat**
   - 当 `output` 中存在 `type:"message"` & `content[].type == "input_image"` 或 `output` 包含 `type:"input_file"`，要映射成 ChatCompletion `message.content` 中的对应结构（例如 `{"type":"image_url","image_url":{"url":...}}`）。
   - 文档输出可映射到 `message.metadata["files"]` 或直接给出下载链接。

### 风险
- 某些后端需要先调用 `/responses/input-items` 上传文件，这要求 forwarder 具有临时存储/上传能力。
- 若项目 B 传入的图片是临时 URL，需要 forwarder 获取数据并按后端需求上传；实现复杂，需要评估。

---

## 5. 观测性 / 自动化验证

### 现状
- DEBUG 日志对每个 delta 均打印，问题排查噪音大，也未提供结构化日志或 request id。
- 缺少自动化回归脚本。

### 计划
1. **日志治理与可观测性**：为 delta 日志增加节流/截断开关，引入结构化日志（JSON）和 request id，必要时接入简单的 `/metrics` 或 Prometheus 导出端点。
2. **回归/脚本**：提供 CLI 或 pytest，将“文本 + reasoning + 工具调用”串起来做端到端校验，确保未来修改不回归。

---

## 实施步骤建议

1. **已完成**：Python SDK streaming + Reasoning 暴露（含 auto include）+ 常见超参透传（logit_bias/logprobs/top_logprobs/seed 等）。
2. **已完成**：Tools/Function 协议映射（含工具响应转换、流式 tool delta 转发）。
3. **规划中**：多模态（图片/文件）、观测性增强、日志降噪。

完成全部里程碑后，forwarder 将提供：
- 默认实时流式 Chat 输出（含 reasoning）。
- 完整的工具 & 多模态互通。
- `/v1/chat/completions` 流式输出与完善的观测性/自动化验证能力。
