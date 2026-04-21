# 后端回复类别说明

本文档说明后端当前如何给用户输入分类，并据此生成不同类型的回复。当前逻辑主要来自 `code/dialogue_policy.py`。

## 一、回复类别

后端会先把用户本轮输入识别成一个 `mode`，不同 `mode` 会使用不同的 system prompt 和生成参数。

| mode | 中文说明 | 典型触发 | 后端回复目标 |
| --- | --- | --- | --- |
| `translation` | 翻译 | 用户说“翻译这句话”“translate this sentence”等 | 翻译最近上下文里被引用的句子，不继续角色扮演 |
| `greeting` | 问候 | 用户只说“你好”“hello”“hi”等，并且当前不在角色扮演中 | 自然打招呼，不把问候变成练习题 |
| `closing` | 收束/结束任务 | 用户说 `that's all`、`already finished`、`done`、`就这些`、`不用了` 等 | 确认当前任务结束，不再继续追问新问题 |
| `scene_setup` | 场景建立 | 用户说“我们练习点餐/订机票/旅行英语”“你扮演服务员”等 | 建立练习场景，进入角色扮演或练习流程 |
| `correction_setup` | 纠错规则设置 | 用户说 `please correct me if...`、`correct my mistakes` 等 | 确认后续会纠错，但不立刻改写这句话 |
| `correction` | 句子纠错 | 用户明确说“纠正”“correct”“fix”“natural”等 | 给出更自然的英文说法和简短中文说明 |
| `explanation` | 用法解释 | 用户问 `why`、`meaning`、`difference`、“为什么”“用法”等 | 用中文解释英语词句或语法用法 |
| `language_switch` | 切换语言 | 用户说“用中文”“用英语”“speak Chinese/English”等 | 按用户要求切换回复语言，同时保留当前上下文 |
| `roleplay` | 角色扮演/普通英语练习 | 当前已有场景，或用户输入主要是英文 | 继续当前场景，回应用户并推动练习 |
| `mixed_practice` | 中英混合练习 | 用户一句话里既有中文又有英文，且不属于其他明确类别 | 按练习上下文处理，必要时解释或继续对话 |
| `general_tutor` | 普通教学兜底 | 不符合以上类别的输入 | 作为英语教练给出通用引导 |

## 二、分类优先级

后端不是随机选择类别，而是按顺序判断。大致优先级如下：

1. 翻译请求：`translation`
2. 简单问候：`greeting`
3. 结束/收束表达：`closing`
4. 新建练习场景：`scene_setup`
5. 停止纠错：根据上下文回到 `roleplay` 或 `general_tutor`
6. 纠错规则设置：`correction_setup`
7. 明确纠错：`correction`
8. 用法解释：`explanation`
9. 语言切换：`language_switch`
10. 已有角色扮演上下文：`roleplay`
11. 纯英文输入：`roleplay`
12. 中英混合输入：`mixed_practice`
13. 兜底：`general_tutor`

## 三、回复状态属性

除了 `mode`，后端还会从历史对话中提取一个简短状态卡，放进 system prompt。主要字段如下：

| 属性 | 中文说明 |
| --- | --- |
| `mode` | 当前回复类别 |
| `roleplay` | 当前是否处在角色扮演场景中 |
| `setup` | 当前场景设置，例如“你扮演机场柜台工作人员” |
| `practice_language` | 当前练习语言，例如 English 或 Chinese |
| `repeat_user_answer` | 是否允许复述用户回答 |
| `teacher_should_ask` | 用户是否要求 AI 主动提问 |
| `correction_mode` | 当前纠错模式，例如关闭纠错 |
| `last_user_english` | 最近一句像英语练习内容的用户输入 |
| `last_assistant_text` | 最近一条 assistant 回复 |
| `recent_ai_questions` | 最近 AI 问过的问题，用于避免重复追问 |
| `user_facts` | 用户在对话中提供的事实信息 |

## 四、生成参数差异

不同类别使用的生成参数也不同，主要控制回复长度和随机性。

| mode | 回复长度倾向 | 随机性倾向 |
| --- | --- | --- |
| `translation` | 中等 | 很低，保证翻译稳定 |
| `greeting` | 短 | 较低 |
| `closing` | 短 | 较低，避免又开始新话题 |
| `language_switch` | 短 | 很低 |
| `correction_setup` | 短 | 较低 |
| `roleplay` | 中短 | 中等偏低 |
| `correction` | 中等 | 很低 |
| `explanation` | 中等偏长 | 较低 |
| `mixed_practice` | 中等偏长 | 较低 |
| `general_tutor` | 中等 | 中等偏低 |

## 五、当前设计目标

当前后端策略的核心目标是：

- 翻译、纠错、解释、角色扮演互不混淆。
- 用户说结束时，后端要真的收束，而不是继续追问。
- 用户让 AI 主动提问时，AI 才主动推动练习。
- 用户要求不要纠错时，后端不应强行进入纠错模式。
- 避免重复上一轮 assistant 的问题或回复。
- 输出尽量保持纯文本，避免 Markdown、标题、粗体、列表等格式。

