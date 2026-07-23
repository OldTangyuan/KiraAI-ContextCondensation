# Context Condensation 上下文压缩插件设计文档

> 版本：0.2.0  
> 日期：2026-07-23  
> 状态：设计阶段

---

## 一、背景与目标

### 1.1 问题

KiraAI 的 `SessionManager` 将对话历史存储在 `data/memory/chat_memory.json` 中，按轮次（chunk）追加，超过 `bot.max_memory_length`（默认 10）后截断旧轮次。每轮对话的 token 量随内容复杂度波动，长工具结果、图片描述等容易撑爆上下文窗口。

### 1.2 目标

在不修改框架核心代码的前提下，通过插件实现：

- **压缩旧对话历史**：将超过锚定区的轮次压缩为信息密度更高的摘要
- **KV-cache 友好**：不碰 `system_prompt`，摘要放在固定位置，增长期只追加末尾
- **分层压缩**：同层内容两两合并，任务单一，避免一次性灌入过多内容导致注意力稀释
- **后台预压缩**：缓存中提前压缩，阈值触发时摘要立即可用
- **精准追踪**：任意摘要可追溯到原始轮次，即使框架截断或压缩滞后也不串号
- **容错降级**：压缩失败时不污染缓存，缓存堆积时触发保底机制

---

## 二、KiraAI 上下文机制（适配前提）

### 2.1 每轮请求的上下文构建流程

```
消息到达 → EventBus → MessageProcessor.handle_im_batch_message()
  │
  ├─ session_memory = SessionManager.fetch_memory(sid)
  │    → 从 chat_memory.json 加载所有 chunk，展平为消息列表
  │    → 超过 max_memory_length 的旧 chunk 已被截断
  │
  ├─ request = LLMRequest(messages=session_memory[:], ...)
  │    → req.messages = 纯历史消息（user/assistant/tool），无 system，无当前输入
  │
  ├─ request.system_prompt.extend(agent_prompt_list)
  │    → 11 个 Prompt section: role/persona/attention/output/format/
  │       accounts/sessions/chat_env/memory/tools/time
  │
  ├─ request.user_prompt.append(Prompt(当前用户消息))
  │
  ├─ 触发 ON_LLM_REQUEST 钩子（按优先级）
  │    → SYS_HIGH:  kira-ai（tag_set + user_prompt 格式化）
  │    → MEDIUM:    chat/memory/session_tools/hippocampus（注入 system_prompt）
  │    → LOW:       file（过滤工具）
  │    → LOW-1:     ★ 本插件（压缩 req.messages）
  │
  ├─ request.assemble_prompt()
  │    → system_prompt 拼接为 messages[0]（system role）
  │    → user_prompt 拼接后追加到 messages 末尾（user role）
  │
  └─ Agent 循环 → LLM 调用 → 回复
```

### 2.2 关键约束

| 约束 | 说明 | 对插件的影响 |
|------|------|-------------|
| **req.messages 无 system** | 钩子执行时 messages[0] 通常是第一条 user 消息 | 插件可以自由操作 messages 列表 |
| **assemble_prompt 后才有 system** | system 在 [0]，user 输入在末尾 | 插件修改的 messages 会被 assemble_prompt 自动包裹 |
| **持久化只存 new_messages** | `update_memory(sid, new_chunk)` 只追加当前轮 | **插件注入的摘要不会被框架持久化**，每轮需重新注入 |
| **框架按 max_memory_length 截断** | 旧轮次从 chat_memory.json 中移除 | 插件缓存需独立保留所有轮次，不依赖框架存储 |
| **一轮 = 一个 chunk** | user + assistant (+ tool) 构成一个原子单元 | 压缩分组以轮次为单位，不拆散 tool_call + tool_result |

---

## 三、核心设计

### 3.1 术语定义

| 术语 | 定义 |
|------|------|
| **轮次（Round）** | 一个完整的对话回合：user 消息 + assistant 回复（+ 可选 tool 调用链）。等同于 KiraAI 的 chunk |
| **锚定区（Anchor）** | 最近 N 轮（默认 5），始终保留原文不压缩。锚定区随对话推进而滑动 |
| **增长区（Growth）** | 锚定区之前的活跃轮次，压缩候选 |
| **摘要（Summary）** | 增长区压缩后的单条 user 消息，覆盖所有已压缩轮次的关键信息 |
| **缓存（Cache）** | 插件维护的独立轮次存储，不限长度，记录所有见过的轮次及压缩状态 |

### 3.2 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  原上下文（req.messages，框架每轮从 chat_memory.json 加载）        │
│  [R1][R2][R3][R4][R5][R6][R7][R8][R9][R10]                      │
│   ←────── 框架 max_memory_length 截断 ──────→                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ON_LLM_REQUEST (LOW-1)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  本插件处理                                                       │
│                                                                   │
│  1. sync(): 用指纹匹配同步 req.messages → 缓存                    │
│  2. 识别已压缩轮次 → 从 req.messages 移除                         │
│  3. 前置摘要（如有）                                               │
│  4. 触发后台压缩（如增长区有新未压缩轮次）                         │
│                                                                   │
│  输出 req.messages:                                               │
│  [摘要user块][R6][R7][R8][R9][R10]                               │
│   ↑ 压缩的      ↑ 锚定区（最近5轮原文）                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ assemble_prompt()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  最终发给 LLM 的消息序列                                          │
│  [system_prompt][摘要user][R6_user][R6_asst]...[R10_asst][当前输入]│
│   ↑ 框架插入    ↑ 插件注入  ↑──────── 原文 ────────↑  ↑ 框架追加  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 摘要的持久化与重注入

**关键设计**：摘要不写入框架的 `chat_memory.json`，而是存储在插件自己的缓存文件中。每轮 `ON_LLM_REQUEST` 钩子执行时重新注入。

原因：
- 框架的 `update_memory()` 只存当前轮的 `new_messages`，不存完整的修改后 `req.messages`
- 如果把摘要写进 `chat_memory.json`，会破坏框架的轮次结构（摘要不是标准 chunk）
- 插件独立存储更灵活，可以随时更新摘要内容而不影响框架

流程：
```
每轮 ON_LLM_REQUEST:
  1. 框架已从 chat_memory.json 加载原始历史 → req.messages
  2. 插件 sync(): 指纹匹配，识别哪些轮次已在缓存中
  3. 插件检查缓存：哪些轮次已被压缩（status=compressed/archived）
  4. 从 req.messages 中移除已压缩的轮次
  5. 前置最新摘要（从缓存读取）
  6. 触发后台压缩（如有新的未压缩增长区轮次）
```

---

## 四、压缩策略

### 4.1 周期循环

```
周期开始（刚完成注入）：
  req.messages = [摘要][锚定区5轮（最近5轮）]
  缓存状态: 摘要就绪, 锚定区 active, 旧增长区 compressed/archived

增长期（每轮追加1轮）：
  第1轮: [摘要][锚定5轮][新轮1]         → 7项
  第2轮: [摘要][锚定5轮][新轮1][新轮2]  → 8项
  ...
  第5轮: [摘要][锚定5轮][新轮1~5]       → 11项（到达 max_memory_length）

  增长期内后台持续压缩新轮1~5

注入触发（活跃轮次达到 max_memory_length）：
  1. 等待后台压缩完成
  2. 将新轮1~5的压缩结果与旧摘要合并 → 新摘要
  3. req.messages = [新摘要][最近5轮原文]
  4. 旧轮次标记为 archived, 被压缩轮次标记为 compressed
  5. 开始新周期
```

**锚定区滑动说明**：

锚定区始终是"当前最近的5轮"。每个周期结束后，锚定区自然滑动到新的最近5轮：

```
周期1结束: [摘要(R1-R10)][R11][R12][R13][R14][R15]    ← 锚定=R11~R15
周期2增长: [摘要(R1-R10)][R11]...[R15][R16]...[R20]
周期2结束: [摘要(R1-R20)][R16][R17][R18][R19][R20]    ← 锚定=R16~R20
周期3增长: [摘要(R1-R20)][R16]...[R20][R21]...[R25]
周期3结束: [摘要(R1-R30)][R21][R22][R23][R24][R25]    ← 锚定=R21~R25
```

摘要覆盖范围随周期扩大，锚定区始终是最新的5轮原文。

### 4.2 分层压缩管道

**核心原则**：同层内容两两合并，绝不混合原始与已压缩内容。每步任务单一，LLM 注意力集中。

```
Layer 0: 原始轮次
  按字数分组（max_chars_per_group，默认800字）
  短轮次合并到一组，长轮次单独一组
  每组 → 1次LLM调用 → 1条一级摘要

Layer 1: 一级摘要
  两两配对合并
  [摘要A + 摘要B] → 1次LLM调用 → 1条二级摘要

Layer 2+: 高级摘要
  继续两两配对，直到只剩1条

最终: 所有增长区轮次 → 1条最终摘要
```

**字数分组规则**：

```
不是固定2条2条，而是按累积字数
组1: [轮次A(80字) + 轮次B(200字)] = 280字 → 压缩
组2: [轮次C(1500字)] = 单独压缩（太长不合并）
组3: [轮次D(100字) + 轮次E(300字)] = 400字 → 压缩
```

### 4.3 后台异步压缩

```
增长区出现新轮次时:
  ├─ 预处理: 若轮次含过长 tool 结果或图片描述 → 缓存中压缩（原上下文不动）
  ├─ 分组: 新轮次按字数分入 layer 0
  ├─ 压缩: 未压缩的 layer 0 分组 → layer 1 摘要
  ├─ 合并: 同层摘要两两配对 → 更高层
  └─ 持久化: 写入缓存文件

整个过程异步执行，不阻塞消息回复。
摘要可能落后1轮，但不影响正确性（下一轮会补上）。
```

### 4.4 注入时的摘要合并

当阈值触发时，需要将**后台压缩产生的新摘要**与**上一周期的旧摘要**合并：

```
旧摘要: "用户之前讨论了浏览器下载问题和矿泉水健康问题..."
新压缩: "用户后来问了米画师私信方法和NapCat版本信息..."

合并 → 新摘要: "用户先后讨论了浏览器下载、矿泉水健康、米画师私信、
              NapCat版本等问题，助手分别给出了解答和建议..."
```

这是1次额外的 LLM 调用，仅在注入时发生（每周期1次）。

### 4.5 压缩失败与保底机制

**核心原则**：压缩失败不污染缓存内容，缓存堆积时触发保底而非继续正常替换。

#### 4.5.1 单次压缩失败

```
某次 LLM 压缩调用失败（模型报错/超时/网络异常）:
  ├─ 该分组保持未压缩状态（compressed=false）
  ├─ 缓存中的轮次和分组记录不受影响
  ├─ 下一轮后台压缩时重试该分组
  └─ 不影响 req.messages 的正常注入逻辑（用已有摘要）
```

#### 4.5.2 连续失败导致缓存堆积

```
缓存轮次数 ≥ 2 × max_memory_length（保底阈值）:
  ├─ 暂停正常替换逻辑（不触发注入）
  ├─ 启动保底压缩：把所有未压缩的增长区轮次一次性合并压缩
  │   （放弃分层，直接一股脑喂给 LLM，优先把缓存降下来）
  ├─ 保底压缩成功 → 恢复正常逻辑
  └─ 保底压缩也失败 → 记录错误日志，等待下一轮重试
```

**保底阈值**：缓存最大长度 = `2 × bot.max_memory_length`。默认 max_memory_length=10 时，缓存最多容纳 20 轮。超过即触发保底。

**保底压缩策略**：不再按字数分组、不再分层，直接将所有未压缩轮次的文本拼接后一次性交给 LLM 压缩。质量不如分层压缩，但能防止缓存无限增长。

---

## 五、缓存与追踪设计

### 5.1 轮次指纹

每条轮次用**首条 user 消息的 content 前200字**做 SHA-256 生成16位指纹。

KiraAI 的用户消息格式包含唯一标记：
```
[Jul 16 2026 23:06 Thu] [message_id: 1204804509] [user_nickname: 老汤圆, user_id: 3521466632] | 不要
```

`message_id` 天然唯一，因此指纹不会碰撞。即使框架截断了旧轮次、上下文位置发生大幅变化，指纹匹配仍能准确识别每条轮次。

### 5.2 轮次状态机

```
                    ┌──────────┐
  新轮次 sync() ──→ │  active  │ ← 在当前 req.messages 中，保留原文
                    └────┬─────┘
                         │ 被分配到压缩分组
                         ▼
                    ┌──────────┐
                    │compressing│ ← 压缩进行中，仍在 req.messages 中
                    └────┬─────┘
                         │ 压缩完成，摘要已生成
                         ▼
                    ┌──────────┐
                    │compressed │ ← 摘要就绪，从 req.messages 移除
                    └────┬─────┘
                         │ 下一周期注入后
                         ▼
                    ┌──────────┐
                    │ archived │ ← 历史归档，不再出现在 req.messages 中
                    └──────────┘
```

### 5.3 存储格式

文件路径：`data/plugin_data/context_condensation/caches/{sid}.json`

```json
{
    "anchor_size": 5,
    "next_round_index": 15,
    "rounds": [
        {
            "i": 0,              // round_index，全局自增，不可变
            "c": 340,            // total_chars
            "p": false,          // is_preprocessed（工具/图片是否已预处理）
            "g": null,           // compression_group（null=未分组, "g0"=分组ID）
            "s": "archived",     // status: active|compressing|compressed|archived
            "fp": "a1b2c3d4e5f6" // fingerprint（内容指纹）
        }
    ],
    "groups": [
        {
            "id": "g0",          // 分组唯一ID
            "l": 1,              // layer: 0=原始轮次分组, 1+=摘要层
            "s": [5, 7],         // source_rounds: 来源轮次的 round_index
            "sg": [],            // source_groups: 来源子摘要的 group_id（layer 2+用）
            "t": "用户遇到浏览器下载卡住的问题...",  // summary_text
            "c": true            // compressed: 是否压缩完成
        },
        {
            "id": "p0",
            "l": 2,
            "s": [],
            "sg": ["g0", "g1"],
            "t": "用户先后讨论了下载问题和矿泉水问题...",
            "c": true
        }
    ]
}
```

### 5.4 追溯能力

任意摘要可通过 `source_rounds` 和 `source_groups` 递归追溯到原始轮次：

```
最终摘要 p1 (layer 3)
├── source_groups: ["p0", "g2"]
│
├── p0 (layer 2)
│   ├── source_groups: ["g0", "g1"]
│   │
│   ├── g0 (layer 1) → source_rounds: [5, 7]  ← 原始轮次
│   └── g1 (layer 1) → source_rounds: [6]     ← 原始轮次
│
└── g2 (layer 1) → source_rounds: [8, 9]      ← 原始轮次

最终: p1 覆盖原始轮次 [5, 6, 7, 8, 9]
```

即使经过多层压缩，仍可精确定位摘要来自哪些原始轮次。

---

## 六、Tool/图片预处理

### 6.1 触发条件

当一条轮次进入增长区（超出锚定区范围）时，检查其中的 tool 结果和图片描述。仅在缓存中预处理，**原 req.messages 不动**。

### 6.2 Tool 结果预处理

Tool 消息的 content 通常是 JSON 字符串（如搜索结果）。处理流程：

```
原始 tool 消息:
  role: tool
  content: '{"url":"...","title":"...","content":"5000字的长文本...","score":0.93}'

处理步骤:
  1. 尝试 JSON 解析 content
  2. 遍历 JSON 中所有字符串字段，累计总字符数
  3. 若总字符数 > tool_result_max_chars（默认2000）:
     a. 提取 JSON 中所有文本内容拼接
     b. 调用压缩 LLM 总结
     c. 用压缩结果替换 JSON 中的文本字段
     d. 添加标记字段 "_condensed": true 表示已被压缩
  5. 重新序列化为 JSON 字符串，替换缓存中的 content

压缩后:
  role: tool
  content: '{"url":"...","title":"...","content":"200字总结...","score":0.93,"_condensed":true}'
```

**标记字段 `_condensed`** 的作用：
- 压缩管道识别到该字段时知道内容已被预处理，不再重复压缩
- 调试时可快速判断哪些 tool 结果被处理过

### 6.3 图片描述预处理

用户消息中的图片描述格式为 `[Image: 描述文本]`。处理流程：

```
原始 user 消息:
  content: "[Jul 17 2026 22:40 Fri] ... | [Image: 这是一张Q版卡通风格的图片，
           描绘了一位正在使用电脑的银发女孩。人物特征：银白色短发...
           （800字描述）]"

处理步骤:
  1. 正则匹配所有 [Image: ...] 块
  2. 对每个图片描述:
     a. 提取描述文本
     b. 若描述长度 > tool_result_max_chars:
        - 调用压缩 LLM 压缩描述
        - 替换原描述文本
  3. 多张图片时逐个处理
  4. 替换缓存中的 content

压缩后:
  content: "[Jul 17 2026 22:40 Fri] ... | [Image: Q版银发女孩用电脑的卡通图，
           疲惫熬夜风格（已压缩）]"
```

**多图片处理**：一条消息中可能包含多个 `[Image: ...]`，每个独立判断长度并独立压缩。只有超过阈值的才处理，短描述保持原样。

### 6.4 原则汇总

- **仅缓存中操作**：原 req.messages 中的 tool 结果和图片描述保持原文
- **轮次完整性**：tool_call + tool_result 作为原子单元，不拆散
- **标记压缩状态**：tool 用 `_condensed` 字段，图片用"（已压缩）"后缀
- **预处理后重新计算字符数**：影响后续压缩分组
- **预处理结果参与压缩**：压缩管道使用预处理后的版本

---

## 七、KV-cache 优化分析

### 7.1 消息序列结构

```
发给 LLM 的最终消息序列（assemble_prompt 后）:

[0] system     ← 框架 system_prompt，永不变化 → KV cache 100%命中
[1] user       ← 摘要块，压缩周期内不变 → KV cache 命中
[2] user       ← 锚定区轮次1的user消息
[3] assistant  ← 锚定区轮次1的assistant消息
...
[N] user/assistant ← 锚定区最后一轮
[N+1] user     ← 当前用户输入（框架追加）
```

### 7.2 周期内缓存命中分析

```
注入后第1轮:  [sys][摘要_v2][R11][R12][R13][R14][R15][当前输入]
                新    新     ←── 全新，无cache ──→

注入后第2轮:  [sys][摘要_v2][R11][R12][R13][R14][R15][R16][当前输入]
                缓存  缓存   缓存  缓存  缓存  缓存  缓存  新    新
                ↑────────── 7条全命中 ──────────↑

注入后第3轮:  [sys][摘要_v2][R11]...[R16][R17][当前输入]
                缓存  缓存   ...   缓存  缓存   新    新
                ↑──────── 8条命中 ────────↑

...

注入后第5轮（再次触发注入）:
              [sys][摘要_v2][R11]...[R20][当前输入]
                缓存  缓存   ...   缓存  新    新
              → 触发压缩 → cache 重置
              [sys][摘要_v3][R16][R17][R18][R19][R20][当前输入]
                缓存  新     ←── 全新 ──→
```

**结论**：周期内（约5轮）KV cache 命中率极高，仅注入那一轮重置。

### 7.3 摘要角色选择

摘要使用 `user` role 而非 `system` role：

| 选项 | 优点 | 缺点 |
|------|------|------|
| **user role**（选用） | 不破坏 system_prompt 的 KV cache | 连续 user 消息可能被 API 合并 → 用前缀标记区分 |
| system role | 语义更准确 | 破坏 system_prompt KV cache |

选用 user role + 前缀标记：`[对话历史摘要 - 以下是你与用户之前对话的关键信息，不是用户当前说的话]`

---

## 八、人设注入（可选）

### 8.1 开关

`schema.json` 中 `use_persona_in_compression` 开关控制是否在压缩时使用人设。

### 8.2 效果

开启后，每次压缩 LLM 调用时注入一条 system 消息：

```
system: 在总结时，请套用以下角色的语气、词汇风格和表述习惯进行总结：
        {人设内容}
user: {压缩提示词}
```

让摘要的语气和风格与角色一致，避免摘要和正文风格割裂。

### 8.3 人设来源

通过 `self.ctx.persona_mgr.get_active_persona()` 获取当前激活的人设文本。

---

## 九、边界情况处理

### 9.1 真实上下文与缓存严重不匹配

**场景**：用户在 WebUI 中清空了某个会话的历史，或框架因异常重置了 `chat_memory.json`。此时 `req.messages` 为空或只有1-2轮，但缓存中可能有几十轮历史。

**检测方式**：sync() 时发现 req.messages 中的轮次数量远少于缓存中的 active 轮次（如缓存有10轮active但 req.messages 只有2轮），判定为严重不匹配。

**处理（由开关控制）**：

| 开关状态 | 行为 |
|----------|------|
| `inject_on_mismatch = false`（默认） | 清空该会话的缓存文件，下次从头开始。最安全，不引入可能过时的摘要 |
| `inject_on_mismatch = true` | 将缓存中所有未压缩轮次一次性压缩为1条摘要，注入到 req.messages 前面。保留历史记忆，但摘要可能包含已过时的信息 |

**开启 `inject_on_mismatch` 时的流程**：
```
1. 检测到严重不匹配（缓存 active 轮次 >> req.messages 轮次）
2. 将缓存中所有 active 且未压缩的轮次一次性合并压缩
3. 与已有摘要（如有）合并
4. 注入到 req.messages 前面
5. 所有旧轮次标记为 archived
6. 缓存以当前 req.messages 的轮次为新的 active 起点
```

### 9.2 框架截断

**场景**：框架 `max_memory_length=10`，对话进行到第12轮，框架只保留第3-12轮。

**处理**：
- 缓存不限长度，保留所有轮次（第1-12轮都在缓存中）
- 第1-2轮虽已从框架存储中移除，但缓存的压缩状态仍有效
- 指纹匹配确保：即使框架加载的轮次变了，缓存仍能正确识别

### 9.3 压缩滞后

**场景**：后台压缩还没完成，但已到达注入阈值。

**处理**：
- 注入前检查 `is_compressing` 标志
- 最多等待30秒（轮询检查）
- 超时则使用已有的部分压缩结果（可能不完整，但不会出错）
- 下一轮补全

### 9.4 压缩失败与缓存堆积

**场景**：压缩 LLM 反复报错，缓存中未压缩轮次持续堆积。

**处理**：见 §4.5 压缩失败与保底机制。

- 单次失败：不污染缓存，下轮重试
- 缓存达到 `2 × max_memory_length`：暂停注入，启动保底一次性压缩
- 保底也失败：记录日志，等待下轮重试

### 9.5 摘要重注入

**场景**：每轮框架从 `chat_memory.json` 加载原始历史，上一轮注入的摘要不在其中。

**处理**：
- 每轮 `ON_LLM_REQUEST` 都从缓存读取最新摘要并注入
- 识别 req.messages 中已被压缩的轮次（status=compressed/archived）并移除
- 确保不会出现"摘要 + 原始轮次"的重复

### 9.6 Tool 调用完整性

**场景**：一个轮次包含 `assistant(tool_calls) → tool(result) → assistant(回复)`。

**处理**：
- 轮次解析时，从 user 消息开始到下一个 user 消息前的所有消息归为一轮
- tool_call + tool_result 天然在同一轮内，不会被拆散
- 压缩时整轮序列化为文本：`[user]: ... [assistant]: ... [tool]: ... [assistant]: ...`

### 9.7 多会话独立

**场景**：多个群/私聊同时活跃。

**处理**：
- 每个会话独立的 `ContextCache` 和 `CompressionPipeline` 实例
- 存储文件按 sid 分离：`caches/{adapter}_{type}_{id}.json`
- 后台压缩任务按 sid 隔离，互不干扰

---

## 十、配置项一览

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | switch | true | 启用/禁用压缩 |
| `anchor_size` | integer | 5 | 锚定区轮数（保留原文的最近N轮） |
| `max_context_rounds` | info | 自动 | 读取自框架 `bot.max_memory_length` |
| `max_chars_per_group` | integer | 800 | 压缩分组最大字符数 |
| `use_persona_in_compression` | switch | false | 压缩时是否使用人设 |
| `compression_model` | model_select | "" | 压缩用 LLM（留空=默认 fast LLM） |
| `preprocess_tool_results` | switch | true | 预处理过长工具结果和图片描述 |
| `tool_result_max_chars` | integer | 2000 | 工具结果/图片描述预处理阈值 |
| `summary_max_chars` | integer | 1500 | 最终摘要最大字符数 |
| `summary_prefix` | text | `[对话历史摘要...]` | 摘要前缀标记 |
| `async_compression` | switch | true | 异步后台压缩 |
| `inject_on_mismatch` | switch | false | 上下文严重不匹配时是否注入缓存摘要（关闭=清空缓存） |
| `debug_log` | switch | false | 调试日志 |

---

## 十一、文件结构

```
data/plugins/context_condensation/
├── manifest.json              ← 插件元信息
├── schema.json                ← WebUI 配置面板
├── requirements.txt           ← 依赖（目前为空）
├── __init__.py
├── main.py                    ← 入口：BasePlugin + ON_LLM_REQUEST 钩子
├── context_cache.py           ← 上下文缓存：指纹匹配、轮次追踪、持久化
├── compressor.py              ← 分层压缩管道：字数分组、同层合并
├── preprocessor.py            ← 预处理：工具结果/图片描述压缩
└── DESIGN.md                  ← 本文档

data/plugin_data/context_condensation/
└── caches/
    ├── Linger_dm_3521466632.json    ← 私聊会话缓存
    └── Linger_gm_115985242.json     ← 群聊会话缓存
```

---

## 十二、待完善项

1. **摘要重注入逻辑**：每轮从缓存读取摘要并注入，移除已压缩轮次的具体实现
2. **管道状态恢复**：插件重启后从缓存文件恢复压缩管道的中间状态
3. **缓存清理**：定期清理 archived 轮次过多或无引用的缓存文件
4. **压缩质量监控**：记录压缩前后的 token 数、信息保留率等指标
5. **并发安全**：多会话同时压缩时的资源隔离和限流
6. **保底压缩实现**：缓存超限时的一次性合并压缩逻辑
7. **不匹配检测算法**：精确判定"严重不匹配"的阈值和逻辑
