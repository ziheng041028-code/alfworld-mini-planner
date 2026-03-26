1、random_baseline.py 已经把 ALFWorld text 环境跑通了，能加载 game.
tw-pddl、读取 admissible_commands，并且搭了一个 random baseline。

- random_baseline.py的核心流程是：
$\text{observation}_t, A_t \xrightarrow{\text{random.choice}(A_t)} a_t \xrightarrow{\text{env.step}(a_t)} \text{observation}_{t+1}, A_{t+1}$

目前可以保存完整 trajectory 和 readable 日志。
随机策略已经验证了环境链路没问题，但在 30 步内无法完成目标导向任务。
我下一步准备做一个 rule-based baseline，再和 random 做小规模对比。

2、