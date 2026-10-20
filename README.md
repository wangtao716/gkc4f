# 别让 AI 学会你

一个会学习玩家走位习惯的二维 Boss 对战游戏。V0.2 支持自由走位、蓄力后松手射击，以及有明确瞄准、锁定和破绽阶段的 Boss 战。你越习惯用同一种方式躲避，Boss 就越有机会提前封锁你的去路；尝试改变方向，让它的预测落空。

当前版本先跑通软件：使用键盘控制，在 CPU 上完成行为学习，不需要独立显卡、网络服务或 myDAQ。课程项目后续再接入 NI myDAQ、摇杆、按钮和 LED。

## 安装与启动

开发环境使用已有的 Conda 环境 **`gkc3C`（Python 3.10.18）**。在项目根目录打开终端：

```powershell
conda activate gkc3C
python -m pip install -r requirements.txt
python main.py
```

也可以使用 `python -m adaptive_boss` 启动。若终端尚未配置 Conda 激活，可直接使用该环境的解释器：

```powershell
& 'D:\ProgramData\Anaconda3\envs\gkc3C\python.exe' -m pip install -r requirements.txt
& 'D:\ProgramData\Anaconda3\envs\gkc3C\python.exe' main.py
```

运行依赖为 `pygame-ce`、`numpy` 和 `scikit-learn`；`pygame-ce` 在代码中仍以 `pygame` 导入。

安装依赖后，也可以双击根目录下的 **`run_game.bat`**。它优先使用上述 `gkc3C` 解释器；找不到时尝试通过 Conda 启动同名环境。

## 操作

| 按键 | 功能 |
| --- | --- |
| `WASD` / 方向键 | 自由移动，支持斜向走位 |
| 按住 `J` / `Z`，再松手 | 蓄力后射击，自动瞄准 Boss；短按为轻射 |
| `Space` / `K` | 向当前移动方向冲刺；未移动时沿最近朝向冲刺 |
| `Enter` | 开始游戏 / 暂停时继续 / 结算后再来一局 |
| `Esc` | 局内暂停；暂停或结算时返回主菜单 |
| `Esc`（主菜单） | 退出游戏 |
| `R` | 局内、暂停或结算时重开 |
| `M` | 开关声音 |
| `C` | 切换固定策略 / 学习策略；主菜单仅切换选择，局内切换并重开 |

菜单、暂停和结算界面的按钮也支持鼠标点击。普通移动没有轨道限制，斜向移动与直线移动速度相同。冲刺仍按上、下、左、右四方向执行，冲刺期间短暂无敌，使用后需要等待冷却。

蓄力约 0.9 秒达到上限，期间移速降为正常的 70%；松手时发射一枚子弹，继续按住不会自动连射。冲刺会取消当前蓄力，需要松开攻击键再重新按住。暂停、切出窗口和离开对局也会取消蓄力，避免恢复时意外开火。

## 玩法与 AI

Boss 交替使用**定点轰击**与**动态横扫**。每轮先瞄准玩家，地面预警在这一阶段跟随位置；进入锁定阶段后，轰击圆心和横扫轨迹固定。横扫会沿预先展示的方向转动，只有生效中的危险区造成伤害。部分回合在主攻击后还有一次单独预警的延迟追击。

一轮攻击和追击全部结束后，Boss 暴露约 1.4 秒破绽。普通阶段有护甲，子弹只造成基础伤害的 28%；破绽阶段伤害为基础的 165%。观察锁定和追击提示，利用走位躲避，在破绽期开火更有效。锁定、追击预警、破绽开启和满蓄力分别有声音提示，可按 `M` 静音。

两种模式使用相同的基础战斗规则：

- **固定策略**：Boss 随机选择策略方向，不使用模型预测。系统仍采集真实走位、训练模型并评估预测，便于观察和对比。
- **学习策略**：积累至少三条完整响应，且预测置信度达到 0.38 时，Boss 使用预测选择横扫方向或后续反制位置；未满足条件时随机选择。两种模式的伤害、阶段时长和移动速度相同。

AI 学习的是相对 Boss 的实际移动方向，普通走位与成功执行的冲刺都参与采样：

| 标签 | 含义 |
| --- | --- |
| `cw` | 顺时针绕行 |
| `ccw` | 逆时针绕行 |
| `in` | 向 Boss 靠近 |
| `out` | 远离 Boss |
| `stay` | 基本停留 |

模型在 CPU 上运行。启用后，预测概率由 75% 的近期平滑频率与 25% 的在线逻辑回归结果组成；逻辑回归使用 `scikit-learn` 的 `SGDClassifier(loss="log_loss")`，输入为瞄准开始时的六项游戏状态和最近三次已完成响应。每条新响应将旧的近期计数乘以 0.72，使新习惯能逐步替代旧习惯。连续三至五轮重复同一种走位就能形成明显偏向，改变走位后也能在几轮内看到变化。

一次学习按以下顺序发生：

1. Boss 开始瞄准时，冻结当时的状态、预测概率和策略方向。瞄准标记随后跟随玩家，锁定时确定最终攻击位置。
2. 从瞄准开始到本轮主攻击及后续追击结束，累计角色相对 Boss 的实际径向、切向移动分量。未执行的冲刺按键，以及贴墙时没有发生的位置变化都不会制造移动样本。
3. 完整窗口中的实际移动路程不足 18 像素时记为 `stay`，否则选择累计移动分量最多的方向；来回移动不会仅因首尾位置重合就被当作停留。先用真实标签检验冻结的预测，再训练模型。战斗结束时未完成的窗口不参与训练。

预测和策略只使用决策前已发生的信息，不读取未来输入；进入锁定后，攻击计划不会因为后续走位而改变。每局重置模型和统计，从头学习。

可以先连续几轮顺时针绕行，再改为逆时针或向外移动，观察预测的变化。预测概率是模型估计，预测正确也不代表反制一定命中。学习统计分别保留最近 20 次的预测准确率和实际反制命中数；方向预测统计排除 `stay`。日志中的历史多数类基线只使用此前的累计标签，用于和快速适应的模型比较。

## 开发与验证

安装开发依赖并运行测试：

```powershell
conda activate gkc3C
python -m pip install -r requirements-dev.txt
python -m pytest
```

常用启动参数：

```powershell
python main.py --mode adaptive --seed 7
python main.py --mode fixed --mute
python main.py --headless --smoke-test --screenshot runs/smoke.png
```

`--mode` 选择初始模式，`--mute` 关闭音效，`--no-logs` 禁用对局日志。`--seed` 固定游戏中的随机数起点，便于在相近条件下调试；玩家输入和操作时机会影响结果，种子相同不代表两局会完全一致。

`--headless` 用于无窗口运行，必须与 `--smoke-test` 一起使用。自动演练执行移动、蓄力射击与冲刺，运行完成后退出；配合 `--screenshot` 保存调试画面。自动演练的模拟时间不等于实际等待时间。

## 代码与记录

| 路径 | 用途 |
| --- | --- |
| `main.py` | 启动入口 |
| `run_game.bat` | Windows 双击启动入口 |
| `adaptive_boss/app.py` | 游戏界面、菜单与主循环 |
| `adaptive_boss/engine.py` | 战斗状态与碰撞、预警、冲刺和响应采集 |
| `adaptive_boss/input.py` | 键盘状态与游戏输入转换 |
| `adaptive_boss/learning.py` | 在线预测、学习和评估统计 |
| `adaptive_boss/audio.py` | 音效生成与播放 |
| `adaptive_boss/telemetry.py` | 每局 JSONL 记录 |
| `adaptive_boss/pilot.py` | 仅根据可见预警操作的自动演练输入 |
| `tests/` | 核心逻辑测试 |
| `requirements.txt` | 运行依赖 |
| `requirements-dev.txt` | 开发与测试依赖 |
| `runs/` | 自动生成的每局 JSONL 记录与调试截图，不纳入 Git |

每局默认生成一份 `runs/session_*.jsonl`，每行是一个 JSON 对象。V0.2 使用 `schema_version: 2`，保留会话信息、冻结的决策、完整窗口的真实响应与预测结果、阶段变化、射击、冲刺、双方受击和结束摘要。可用于比较模式和整理课程实验数据；记录时间使用 UTC。模型当前不跨局保存，重新开始会保留已有日志，并启动一份新记录。

分析时可以通过 `attack_id` 关联 `decision`、`observation`、`phase` 与 `player_hit`。`observation.correct` 表示方向预测是否正确；`player_hit.is_counter` 表示是否被后续反制实际击中，`using_prediction` 表示该轮是否应用了学习策略。结束事件中的 `counter_hits / counter_attempts` 为所有实际反制的命中数与生效次数，`adaptive_counter_hits / adaptive_counter_attempts` 仅统计应用预测的回合。没有反制生效时分母为零，不应据此计算命中率。

会话中的 `input_source` 区分 `keyboard`（真人键盘操作）与 `smoke_test`（自动演练）。整理真人对照实验时，仅选用 `keyboard` 记录，避免混入测试程序的操作。

输入通过 `InputState` 与战斗逻辑分离。后续接入 myDAQ 时，可将摇杆和按钮读数转换为相同的输入状态，复用现有游戏和学习模块；本版不安装 NI-DAQmx 驱动，也不读取硬件。

## 项目资料

- [开题报告](基于myDAQ的自适应AI对战游戏.docx)
- [myDAQ 端口说明](myDAQ端口说明.md)
- [NI myDAQ 使用手册](<1-NI MyDAQ guide.pdf>)


