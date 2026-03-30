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

3、修改 rule_based_baseline 后，agent 在遇到 alarmclock 后 stage 会从 search_both 切换为 search_tool, 但遇到 desklamp 后有会出现在 use desklamp 这一步自旋且跳出自旋后又会局部动作空间里在找最高分垃圾动作导致最终任务无法完成。具体改动如下表:

| 模块  | 原版 | 改进版 | 作用 |
| ---- | --- | ----- | ---- |
| 任务理解 | 只抽关键词 `alarmclock/desklamp` | `parse_task()` 解析成 `verb=examine, target=alarmclock, tool=desklamp`                                                         | 从“词匹配”升级到“结构化任务”                  |
| 策略状态 | 没有阶段                        | `infer_stage()`：`search_both / search_target / search_tool / solve_with_tool`                                               | 不再一条规则打天下                         |
| 记忆   | 只记 `visited_locations`      | 新增 `seen_entities`、`examined_entities`                                                                                      | 知道“目标/工具见过没”“这个东西 examine 过几次”    |
| 重复控制 | `recent_repeat(-6)` 很弱      | `same_as_last(-14)`、`recent_repeat(-8)`、`triple_repeat_risk(-18)`                                                           | 明显增强反自旋能力                         |
| 停滞处理 | 没有                          | `stagnation_count` + `break_stagnation_go(+16)`                                                                             | Observation 不变时强制探索               |
| 任务特化 | 只有“with 时 prefer use”       | 增加 `search_tool_hit(+28)`、`already_have_target_keep_searching_tool(-18)`、`avoid_taking_target_for_examine_with_tool(-12)` 等 | 更贴合 `examine X with Y`            |
| 工程运行 | 容易受当前目录影响                   | 增加 `PROJECT_ROOT` / `sys.path` 处理                                                                                           | 缓解 `No module named src` 这类运行路径问题 |

- 总结下来，baseline 从 “无状态启发式” 推进到了 “有状态但收尾很弱的启发式规划器”。下一步需要把 solve_with_tool 真正做成一个子计划。
