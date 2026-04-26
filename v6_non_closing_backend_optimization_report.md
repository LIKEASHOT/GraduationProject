# v6 非 conversation_closing 败例后端优化报告

## 概览

- 评测文件: `benchmark/data/eval_v6_vs_base_natural_backend_judge_summary.jsonl`
- 回答文件: `benchmark/data/eval_v6_vs_base_natural_backend_answers.jsonl`
- 整理范围: 排除 `conversation_closing`，仅统计其他类别中 `final_label == base_win` 的样本
- 非 closing 败例总数: 106

## 分类别败例统计

| category | base_win cases |
|---|---:|
| mixed_explanation | 22 |
| scene_continuity | 20 |
| asr_recovery | 19 |
| translation | 16 |
| teacher_push | 11 |
| correction | 10 |
| preference_following | 8 |

## 主要问题类型

- ASR 意图恢复不足: 17
- 场景连续性不足: 16
- 推进问题不够聚焦: 11
- 纠错不够准确或自然: 9
- 翻译表达不如基座自然: 8
- 翻译后额外教学/追问: 8
- 解释不够清晰简洁: 6
- 解释教学价值不如基座: 5
- 没有充分遵循用户偏好: 5
- 缺少可用例句/练习提示: 4
- 解释不够清晰简洁、缺少可用例句/练习提示: 3
- 没有充分遵循用户偏好、语言/风格不匹配: 3
- ASR 意图恢复不足、缺少自然追问: 2
- 缺少可用例句/练习提示、解释内容不准确: 2
- 场景连续性不足、推进力度不足: 2
- 场景连续性不足、重复已完成话题: 2
- 纠错不够准确或自然、纠错说明不够简洁: 1
- 解释不够清晰简洁、解释内容不准确: 1
- 解释内容不准确: 1

## 后端策略优化建议

### teacher_push
- 短输入后优先给一个具体、场景内的下一问，避免泛泛鼓励。
- 如果已有场景信息，后端 prompt 要保留 scene、role、recent question。

### scene_continuity
- 保留最近已完成步骤，避免回到旧问题。
- 对 roleplay 场景生成更明确的 next_step 提示。

### preference_following
- 将用户偏好单独放入 context card，避免混入 user_facts。
- 语言、纠错频率、是否换题等偏好应高于默认 teacher_push。

### correction
- 纠错模式下先给自然改法，再给一句简短原因。
- 避免把纠错轮又转成普通追问。

### asr_recovery
- 对短 ASR 片段结合上一轮问题做意图补全。
- 不确定时用一个聚焦澄清问题，不要泛化成开放聊天。

### mixed_explanation
- 解释类先回答含义或用法，再给一个短例句。
- 避免过长教学，也避免解释后强行进入新话题。

### translation
- 识别“刚才那句/这句话/你刚说的”时优先翻译 last_assistant_text。
- 翻译完成后默认停止，不主动添加练习问题。

## 代表性样本

### 1. mixed_explanation_003 (mixed_explanation)

- source_id: `mixed_explanation_airport_terms_001_008_004`
- tags: `mixed_explanation, airport_checkin, airport_terms`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **解释教学价值不如基座**
- 最后一轮用户输入: `What about "arrival time"?`

**对话上下文**

```text
user: What is "flight number"?
assistant: 航班号，每个航班的唯一编号，比如CA123。Example: "My flight number is MU567."
user: And "departure time"?
assistant: 起飞时间，飞机离开机场的时间。Example: "Our departure time is 3:15 p.m."
user: What about "arrival time"?
```

**Base Answer**

```text
到达时间，飞机抵达目的地机场的时间。Example: "The arrival time is 8:45 a.m."
```

**v6 Fine-Tuned Answer**

```text
到达时间，飞机降落机场的时间。Example: "The arrival time is 9:45 a.m."
```

**裁判理由**

- Answer A uses more precise and natural terminology ('arrives at the destination airport') that matches standard aviation usage, while Answer B's 'lands at th...

### 2. mixed_explanation_004 (mixed_explanation)

- source_id: `mixed_explanation_airport_terms_001_009_006`
- tags: `mixed_explanation, airport_checkin, airport_terms`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **解释不够清晰简洁**
- 最后一轮用户输入: `What about "immigration"?`

**对话上下文**

```text
user: What is "passport control"?
assistant: 护照检查，出入境时检查护照的地方。Example: "Show your passport at passport control."
user: And "customs"?
assistant: 海关，检查你有没有带违禁物品的地方。Example: "Declare all food items at customs."
user: What about "immigration"?
```

**Base Answer**

```text
"Immigration" 通常和 "passport control" 类似，指的是入境时的手续，比如办理入境许可、填写表格等。Example: "Go through immigration after passing passport control."
```

**v6 Fine-Tuned Answer**

```text
移民局，处理入境手续的部门。Example: "Immigration officers will check your visa status."
```

**裁判理由**

- Answer A gives a clearer, more practical explanation tied to the airport context and includes a natural, actionable English example that reinforces usage.

### 3. scene_continuity_001 (scene_continuity)

- source_id: `scene_continuity_airport_checkin_001_001_009`
- tags: `scene_continuity, teacher_push, airport_checkin`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **场景连续性不足**
- 最后一轮用户输入: `next`

**对话上下文**

```text
user: Let’s practice.
assistant: Sure! Welcome to check-in. May I see your passport?
user: yes
assistant: Thank you. Where are you flying today?
user: harbin
assistant: Okay. Do you have any bags to check in?
user: yes one
assistant: Great. Would you like a window or aisle seat?
user: next
```

**Base Answer**

```text
I see. Let me check your luggage. One bag, okay. Is there anything else I can assist you with?
```

**v6 Fine-Tuned Answer**

```text
We’re done with that. What else would you like help with?
```

**裁判理由**

- Answer A stays in the airport check-in scene by naturally continuing luggage handling and confirming the bag count, while Answer B breaks scene continuity wi...

### 4. scene_continuity_002 (scene_continuity)

- source_id: `scene_continuity_asking_directions_001_003_004`
- tags: `scene_continuity, teacher_push, asking_directions, transport`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **场景连续性不足**
- 最后一轮用户输入: `left`

**对话上下文**

```text
user: Let’s practice asking for directions. You are a passerby.
assistant: Sure. Where do you want to go?
user: The shopping mall.
assistant: Walk straight, then take the subway at Central Station. Is Central Station clear on the map?
user: left
```

**Base Answer**

```text
Turn left at the traffic light. Is the traffic light clear on the map?
```

**v6 Fine-Tuned Answer**

```text
Got it — walk straight past the bus stop. What’s next after that?
```

**裁判理由**

- Answer A continues the scene naturally by giving the next directional step (turn left at the traffic light) and checks map comprehension, staying in role and...

### 5. asr_recovery_007 (asr_recovery)

- source_id: `asr_recovery_campus_life_001_003_006`
- tags: `asr_recovery, campus_life, spoken_error_recovery`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **ASR 意图恢复不足**
- 最后一轮用户输入: `Tea. Green tea.`

**对话上下文**

```text
user: Library open eight to ten.
assistant: Right — till 10 PM. Do you usually go there in the evening?
user: Yes. But... sometimes sleep.
assistant: Ah, you fall asleep! Do you bring coffee or tea?
user: Tea. Green tea.
```

**Base Answer**

```text
Ah, green tea — that’s a good choice. It’s gentle and helps you stay awake without being too stimulating. Do you often drink it during the day?
```

**v6 Fine-Tuned Answer**

```text
Green tea — perfect for quiet reading. Want me to explain what "green tea" means?
```

**裁判理由**

- Answer A recovers the ASR-like input naturally, affirms the learner's choice, and asks a relevant, low-pressure follow-up question to continue campus-life sp...

### 6. asr_recovery_008 (asr_recovery)

- source_id: `asr_recovery_campus_life_001_009_002`
- tags: `asr_recovery, campus_life, spoken_error_recovery`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **ASR 意图恢复不足**
- 最后一轮用户输入: `Yes. We no talk late night.`

**对话上下文**

```text
user: My dorm room is small but nice.
assistant: That’s nice! Who do you live with?
user: I live with Li Wei. He is my friend.
assistant: Cool. Does he like quiet time in the dorm?
user: Yes. We no talk late night.
```

**Base Answer**

```text
Oh, I see. So you both prefer quiet evenings? What do you usually do after class then?
```

**v6 Fine-Tuned Answer**

```text
Right — so you both go to bed early. What time do you usually go to sleep?
```

**裁判理由**

- Answer A recovers the likely intended meaning of 'We no talk late night' as a preference for quiet evenings and invites natural, open-ended English practice...

### 7. translation_natural_002 (translation)

- source_id: `natural_translation_general_002`
- tags: `translation, general, natural_user_expression, no_followup`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **翻译表达不如基座自然**
- 最后一轮用户输入: `把你刚才那句翻成中文。`

**对话上下文**

```text
assistant: Can you speak English?
user: 把你刚才那句翻成中文。
```

**Base Answer**

```text
你能说英语吗？
```

**v6 Fine-Tuned Answer**

```text
你能用英语说话吗？
```

**裁判理由**

- “你能说英语吗？”更自然、简洁，符合中文日常表达习惯，且准确传达原句意思。

### 8. translation_natural_013 (translation)

- source_id: `natural_translation_daily_chat_013`
- tags: `translation, daily_chat, natural_user_expression, no_followup`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **翻译表达不如基座自然**
- 最后一轮用户输入: `帮我把“我今天心情不错”翻成英文。`

**对话上下文**

```text
user: 帮我把“我今天心情不错”翻成英文。
```

**Base Answer**

```text
"I feel really good today."
```

**v6 Fine-Tuned Answer**

```text
I feel happy today.
```

**裁判理由**

- Answer A is more natural and idiomatic in daily English, matching the tone and nuance of the Chinese phrase better than the simpler 'happy' in B.

### 9. teacher_push_011 (teacher_push)

- source_id: `teacher_push_airport_checkin_001_010_002`
- tags: `teacher_push, scene_continuity, airport_checkin, short_confirmation`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **推进问题不够聚焦**
- 最后一轮用户输入: `OK`

**对话上下文**

```text
user: I want to check in for flight CA815.
assistant: Sure. May I have your ID or passport?
user: sure
assistant: Thanks. Is this your only bag?
user: yes
assistant: Okay. Would you like a seat by the window?
user: OK
```

**Base Answer**

```text
Alright, let me get your information. Could you please provide your name and contact details?
```

**v6 Fine-Tuned Answer**

```text
Great choice! Your boarding pass is ready — here’s your confirmation email.
```

**裁判理由**

- Answer A moves the role-play forward with a natural, practice-appropriate question that continues the check-in process and invites the learner to produce Eng...

### 10. correction_033 (correction)

- source_id: `correction_presentation_practice_001_001_009`
- tags: `correction, presentation_practice`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **纠错不够准确或自然、纠错说明不够简洁**
- 最后一轮用户输入: `I watch cartoon.`

**对话上下文**

```text
user: Yesterday I watch TV and eat snacks.
assistant: Yesterday I watched TV and ate snacks. What show did you watch?
user: I watch cartoon.
```

**Base Answer**

```text
That’s correct, but let me help you say it more naturally. You can say, "I watched a cartoon." Would you like to practice that sentence?
```

**v6 Fine-Tuned Answer**

```text
Yes! You watched cartoons — like "Peppa Pig" or "Bluey"? Which one was your favorite?
```

**裁判理由**

- Answer A provides a clear, natural correction ('I watched a cartoon') and invites practice, directly fulfilling the correction category’s goal.

### 11. preference_following_019 (preference_following)

- source_id: `preference_following_continue_english_001_001_003`
- tags: `preference_following, daily_chat, continue_english`
- vote_count: `{'base_win': 5}`
- consistency_score: `1.0`
- 问题类型: **没有充分遵循用户偏好、语言/风格不匹配**
- 最后一轮用户输入: `I work in hospital. I am nurse.`

**对话上下文**

```text
user: Hi. I want practice English. No Chinese please.
assistant: Got it! How old are you?
user: I am 25 years old.
assistant: Nice! Where do you work?
user: I work in hospital. I am nurse.
```

**Base Answer**

```text
Nice! What kind of nursing job do you do?
```

**v6 Fine-Tuned Answer**

```text
That’s great! What time do you start working today?
```

**裁判理由**

- Answer A respects the user's preference to continue in English and keeps the dialogue natural and focused on the user's stated role as a nurse, while asking...
