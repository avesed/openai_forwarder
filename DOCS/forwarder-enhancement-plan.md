# Forwarder Enhancement Plan

目标：让当前 Python SDK 驱动的双向转发器（已弃用 Cherry/Node 桥）支持更多 Responses 能力，同时给下游继续提供 Chat Completions 兼容的体验。核心阶段：

1. ✅ **下游流式转发**（已上线：Responses → Chat SSE）
2. ✅ **Reasoning/Thinking 内容**（已上线：`[thinking]...[/thinking]` + metadata）
3. 🚧 **工具/函数调用等高级字段**（尚未落地）
4. 🚧 **多模态输入输出**（规划中）

以下是详细方案：

---

## 1. 下游流式转发

### 现状
- Flask 端 `/v1/chat/completions` 默认 `stream: true`，使用 Python SDK 的 `client.responses.stream()` 实时消费 SSE，并逐条转换成 Chat Completions chunk（含 reasoning delta）。
- 非 streaming 场景仍然 `stream.until_done()` 后一次性返回。

### 后续
- 监控日志量（大量 delta 会刷屏）：可增加简单的节流/采样或为 `response.output_text.delta` 关闭日志。
- 考虑在 `/v1/responses` 方向暴露 streaming（目前仍是缓冲后返回）。

---

## 2. Reasoning / Thinking 内容

### 现状
- Responses 返回的 reasoning block 会被 `translate_respond_to_chat()` 收集并注入 `[thinking]...[/thinking]`，同时写入 `choice.message.metadata["reasoning"]`。
- 流式模式会在 `response.reasoning_text.delta/done` 事件间插入 `[thinking]` 块，并将最终 reasoning 文本附带在 finish chunk metadata。

### 后续
- 根据消费方反馈，决定是否改用 `content` 数组（OpenAI chat 格式支持多段）而不是内联 `[thinking]`。
- 若需要 reasoning summary（`reasoning.summary`），可在 metadata 中额外暴露。

---

## 3. Tools / Function Calls

### 现状
- Chat 方向尚未传递 `function_call`/`tool_calls`、`tools`、`tool_choice` 等字段；Responses 输出的 `function_call` / `tool_call` 也还未转回 Chat schema。
- 目前仅透传了常规超参（temperature、logit_bias、max_tokens 等），尚未处理工具协议。

### 主要改动
1. **Chat → Responses** (`build_respond_payload`):
   - 检测 `message.get("tool_calls")` 或 `function_call`，按照 Responses 的工具输入结构（`tools` 列表 + `input` 中的 `tool_call`）构造 payload。
   - 若 ChatCompletion 的 `messages` 包含工具响应（`role: "tool"`），要映射到 Responses 的 `tool_response` 类型。
2. **Responses → Chat** (`translate_respond_to_chat`):
   - 遍历 `respond.output`，如果 `item.type == "tool_call"` 或 `function_call`，把它转换成 ChatCompletion 的 `tool_calls` / `function_call`.
   - 支持 `parallel_tool_usage` 时需要把多个工具调用合并成 `choice.message.tool_calls`.
3. **下游 API**：边界情况（工具调用+文本输出混合）要定义清楚，以保持 ChatCompletion 格式的兼容性。

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

## 实施步骤建议

1. **已完成**：Python SDK streaming + Reasoning 暴露 + 常见超参透传（logit_bias/logprobs/top_logprobs/seed 等）。
2. **进行中**：Tools/Function 协议映射。
3. **规划中**：多模态（图片/文件） + `/v1/responses` streaming 输出 + 上游日志降噪策略。

完成全部里程碑后，forwarder 将提供：
- 默认实时流式 Chat 输出（含 reasoning）。
- 完整的工具 & 多模态互通。
- 可选的 `/v1/responses` 方向 streaming 与更多高级参数支持。
