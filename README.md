# 别让 AI 学会你

一个会学习玩家闪避习惯的二维 Boss 对战游戏。你越习惯用同一种方式躲避，Boss 就越有机会提前封锁你的去路；尝试改变方向，让它的预测落空。

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
| `WASD` / 方向键 | 移动 |
| 按住 `J` / `Z` | 持续射击，自动瞄准 Boss |
| `Space` / `K` | 向当前移动方向冲刺；未移动时沿最近朝向冲刺 |
| `Enter` | 开始游戏 / 暂停时继续 / 结算后再来一局 |
| `Esc` | 局内暂停；暂停或结算时返回主菜单 |
| `Esc`（主菜单） | 退出游戏 |
| `R` | 局内、暂停或结算时重开 |
| `M` | 开关声音 |
| `C` | 切换固定策略 / 学习策略；主菜单仅切换选择，局内切换并重开 |

菜单、暂停和结算界面的按钮也支持鼠标点击；点击主菜单的模式卡后，再点击开始。冲刺按上、下、左、右四方向执行，冲刺期间短暂无敌，使用后需要等待冷却。观察地面预警，在安全位置持续射击，击败 Boss 即可获胜。

## 玩法与 AI

原型包含一个竞技场、一个 Boss、锁定轰击、定向光束和交叉封锁三种基础攻击，以及血量、冲刺冷却、暂停和结算界面。右侧面板展示玩家响应预测和学习统计。

两种模式使用相同的基础战斗规则：

- **固定策略**：Boss 随机选择反制方向，不使用模型预测。系统仍记录响应并评估模型，便于观察和对比。
- **学习策略**：积累至少八条完整响应，且预测置信度达到 0.38 时，Boss 使用预测调整反制危险区的位置；未满足条件时随机选择。学习不会增加伤害或移动速度。

AI 使用 `scikit-learn` 的 `SGDClassifier(loss="log_loss")`，在 CPU 上进行增量逻辑回归。模型结合预警开始时的游戏状态与最近三次真实响应，预测本轮观察窗口中的首次有效冲刺方向，或“未冲刺”。前八条响应使用平滑频率估计，收集足够样本后启用在线模型。

一次学习按以下顺序发生：

1. Boss 开始预警时，保存当时的状态、预测概率和危险区位置。
2. 从预警开始至危险区生效结束，记录第一次成功执行的冲刺方向；冷却期间未执行的按键不计入。
3. 观察窗口完整结束后，没有有效冲刺则记为“未冲刺”；用真实响应检验之前的预测，再更新模型。战斗结束时尚未结束的观察窗口不参与训练。

预警一旦出现，危险区不会追随玩家后续按键改变。预测只使用之前已经发生的信息，不读取玩家未来的输入。每局重置模型和统计，从头学习。

可以先连续几轮向同一侧冲刺，再换方向，观察预测的变化。面板中的概率表示模型估计；它不保证下一次预测正确，也不保证学习策略一定比固定策略更强。面板显示最近 20 次响应的命中率；日志还保存历史多数类基线的预测与命中情况，可据此评估模型，也可筛除“未冲刺”后单独分析方向预测。

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

`--headless` 用于无窗口运行，必须与 `--smoke-test` 一起使用。自动演练运行至约 40 秒模拟时间或对局结束后退出；配合 `--screenshot` 可保存收集至少九条响应后的预警画面。模拟时间不等于实际等待时间。

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
| `tests/` | 核心逻辑测试 |
| `requirements.txt` | 运行依赖 |
| `requirements-dev.txt` | 开发与测试依赖 |
| `runs/` | 自动生成的每局 JSONL 记录与调试截图，不纳入 Git |

每局默认生成一份 `runs/session_*.jsonl`，每行是一个 JSON 对象，包含会话信息、冻结的决策、完整窗口的真实响应与预测结果，以及结束摘要。可用于比较模式和整理课程实验数据；记录时间使用 UTC。模型当前不跨局保存，重新开始会保留已有日志，并启动一份新记录。

会话中的 `input_source` 区分 `keyboard`（真人键盘操作）与 `smoke_test`（自动演练）。整理真人对照实验时，仅选用 `keyboard` 记录，避免混入测试程序的操作。

输入通过 `InputState` 与战斗逻辑分离。后续接入 myDAQ 时，可将摇杆和按钮读数转换为相同的输入状态，复用现有游戏和学习模块；本版不安装 NI-DAQmx 驱动，也不读取硬件。

## 项目资料

- [开题报告](基于myDAQ的自适应AI对战游戏.docx)
- [myDAQ 端口说明](myDAQ端口说明.md)
- [NI myDAQ 使用手册](<1-NI MyDAQ guide.pdf>)


