1、random_baseline.py 已经把 ALFWorld text 环境跑通了，能加载 game.
tw-pddl、读取 admissible_commands，并且搭了一个 random baseline。

- random_baseline.py的核心流程是：
$\text{observation}_t, A_t \xrightarrow{\text{random.choice}(A_t)} a_t \xrightarrow{\text{env.step}(a_t)} \text{observation}_{t+1}, A_{t+1}$

目前可以保存完整 trajectory 和 readable 日志。
随机策略已经验证了环境链路没问题，但在 30 步内无法完成目标导向任务。
我下一步准备做一个 rule-based baseline，再和 random 做小规模对比。

2、rule_based_baseline.py 实现了规则的限制，action, selector_info = select_action(...) 引入了一个显式动作选择器；select_action() 会抽出 examine the alarmclock with the desklamp 中大致这两个关键词：{alarmclock,desklamp}，之后对每个候选动作进行答分，规则如下:

- 动作里包含任务关键词，加分
- 如果 observation 已经提到了目标物体，再加分
- use / examine / take 给基础优先级
- help / inventory / look 降权
- 去没访问过的位置，加分
- 最近重复动作，减分

最后从最高分动作里选一个，如果有多个并列第一，就随机挑一个。
但现在的问题是从 16 步开始反复出现 examine alarmclock@(-0.82,+0.86,-1.36) 这一高分动作，需要进一步修改。
