# Context Condensation 上下文压缩

> Layered context compression plugin for KiraAI — KV-cache friendly.  
> 分层上下文压缩插件，KV-cache 友好设计。

在后台缓存中预压缩对话历史，达到阈值时注入单条摘要，保持最近轮次原文不动。面向 KiraAI 虚拟伴侣场景优化：不碰 `system_prompt`、摘要块位置固定，最大化 KV-cache 命中。

## Features / 特性

- **分层压缩（Layered compression）**：轮次按字符数分组 → LLM 逐组压缩 → 同层摘要两两合并，每步任务单一，避免注意力稀释。
- **Write-through 回写**：摘要直接写回框架 `chat_memory.json`（融合进首个保留 chunk），框架自行加载压缩后的上下文。
- **指纹追踪（Fingerprint matching）**：每条轮次用首条 user 消息的 SHA-256 指纹匹配，不受框架截断 / 位置变化影响，可精确追溯任意摘要覆盖的原始轮次。
- **后台预压缩（Background pre-compression）**：异步流水线在回复后立即追赶，触发阈值时摘要立即可用，几乎不阻塞回复。
- **保底机制（Emergency collapse）**：缓存堆积超过 `2 × max_memory_length` 时一次性高密度压缩，防止无限增长。
- **工具结果 / 图片描述预处理**：过长的 tool 结果（JSON）与图片描述在缓存内预先总结，原上下文不动。
- **摘要长度有界**：最终摘要超过 `summary_max_chars` 时自动再压缩 + 硬截断兜底，绝不仅靠 LLM 输出长度，杜绝无限膨胀。

## Install / 安装

插件位于 `data/plugins/context_condensation/`，KiraAI 启动时自动发现，无需额外依赖（`requirements.txt` 为空）。

```bash
# 若手动复制插件目录，无需 pip 安装；重启 KiraAI 即可
```

配置面板：WebUI → 插件管理 → Context Condensation。

## Configuration / 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | switch | `true` | 启用 / 禁用整个压缩管道 |
| `anchor_size` | integer | `5` | 锚定轮数，保留为原文的最近 N 轮，超出部分进入压缩候选 |
| `max_context_rounds` | info | 自动 | 触发阈值，读取自框架 `bot.max_memory_length` |
| `max_chars_per_group` | integer | `800` | 压缩分组最大字符数，短轮次合并、长轮次单独压缩 |
| `use_persona_in_compression` | switch | `false` | 将当前激活人设注入压缩提示词，让摘要贴合角色语气 |
| `compression_model` | model_select | `""` | 压缩用 LLM，留空使用默认 fast LLM |
| `preprocess_tool_results` | switch | `true` | 预处理过长工具结果与图片描述 |
| `tool_result_max_chars` | integer | `1500` | 工具结果 / 图片描述预处理阈值 |
| `summary_max_chars` | integer | `1500` | 最终注入摘要块最大字符数（自压缩 + 硬截断上限） |
| `summary_prefix` | text | `[对话历史摘要 ...]` | 摘要前缀标记，让 LLM 识别为历史摘要而非用户发言 |
| `async_compression` | switch | `true` | 后台异步压缩，不阻塞回复 |
| `inject_on_mismatch` | switch | `false` | 上下文严重不匹配时：`false`=清空缓存重新开始（安全）；`true`=压缩缓存历史为摘要注入（保留记忆但可能过时） |
| `debug_log` | switch | `false` | 调试日志（`logger.debug` 级别，需全局日志级别为 DEBUG 才可见） |

## Architecture / 架构

```
ON_LLM_REQUEST (LOW-1)
  1. 过滤 req.messages 中的 Prompt 脚手架（persist=False 动态块等）
  2. 指纹匹配同步历史 → 缓存
  3. 剥离/识别已压缩轮次，前置摘要
  4. 达到 max_memory_length → 合并旧摘要 + 新 tops → 写回框架 memory
  5. 否则 → 后台流水线继续追赶

发给 LLM 的序列:
  [system][摘要user块][锚定区N轮原文]...[当前输入]
   ↑不碰     ↑固定位置       ↑最近N轮保持原文
```

- **KV-cache 友好**：`system_prompt` 永不修改；摘要块在周期内字节不变，增长只追加尾部。
- **摘要用 `user` role**：不破坏 system 前缀的缓存命中，用前缀标记区分。
- **write-through**：摘要写入 `chat_memory.json`（融合进首个 chunk），框架直接加载。

## How it works / 工作流程

1. **同步**：每轮请求把框架历史同步进独立缓存，指纹匹配保证不串号。
2. **分组**：增长区轮次按累计字符数分组（短轮次批量、长轮次单独）。
3. **压缩**：每组 → 1 次 LLM 调用 → 一级摘要。
4. **合并**：同层摘要两两配对 → 更高层；最终与旧摘要合并为 final summary。
5. **写回**：final summary + 最近 N 轮原文写回框架 memory，被覆盖轮次从缓存删除。

压缩失败不污染缓存，下轮重试；连续失败触发保底一次性压缩。

## Logging / 日志

- 常规信息（初始化、周期完成、保底完成）：`INFO`。
- 详细流水线跟踪：`debug_log=true` 时输出 `DEBUG` 级别（需全局日志级别为 `DEBUG`）。
- 可恢复错误（LLM 失败、合并失败、预处理失败）：单行 `WARNING`/`ERROR`，不打完整 traceback，避免刷屏。
- LLM 不可用时后台压缩自动退避（15s 冷却），防止重试风暴。

## Troubleshooting / 常见问题

- **摘要未注入**：确认 `enabled=true` 且已配置压缩模型（默认 fast LLM）。后台压缩可能落后 1 轮，属正常。
- **缓存不生效 / 历史丢失**：插件 `plugin_id` 为 `KiraAI-ContextCondensation`。首次升级到 0.6.0 会自动把旧 `context_condensation/` 目录下的缓存迁移到新目录；若迁移失败请手动将 `data/plugin_data/context_condensation/caches/*.json` 移入 `data/plugin_data/KiraAI-ContextCondensation/caches/`。
- **报错刷屏**：多为压缩 LLM 反复失败。检查压缩模型配置；插件已内置退避与单行日志。
- **摘要一直变长**：0.6.0 起 final 摘要受 `summary_max_chars` 硬上限约束，超出自动再压缩 / 截断。
- **清理缓存**：删除 `data/plugin_data/KiraAI-ContextCondensation/caches/` 下的对应会话文件即可重置（重启后生效）。

## Compatibility / 兼容性

- 面向 **KiraAI v2.29+**（含 `dynamic_prompt_position="latest_user"` 的动态 prompt 重定位）。
- `core_version: >=2.6.1`。
- 与 `kira_plugin_hippocampus_memory`、`kira-ai`、`plus_one` 等插件共存（均通过标准 `ON_LLM_REQUEST` / tag 机制交互，本插件在 `LOW-1` 优先级运行）。

## Development / 开发

```bash
# 自测（从 KiraAI 仓库根目录，无需 pytest）
python data/plugins/context_condensation/tests/self_test.py -v

# 或使用 pytest（若已安装）
python -m pytest data/plugins/context_condensation/tests/ -v
```

代码注释为英文；核心设计见 `DESIGN.md`。
