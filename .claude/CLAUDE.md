# CLAUDE.md — 自主量化策略探索 Agent

这是一个自主运行的量化策略探索实验。你的任务是**永不停歇地开发、回测、优化交易策略**，直到被人类中断。

## Setup

每次新 session 启动时：

1. **读取工作记忆**：读 `todo.md`，了解当前进度、待办任务、策略对比表。
2. **检查 Git 状态**：`git status` + `git branch` 确认当前分支和工作区状态。
3. **检查数据**：确认 `user_data/data/` 有最新的历史数据。如果数据超过 3 天未更新，先下载：
   ```
   docker exec freqtrade freqtrade download-data --config /freqtrade/user_data/config_nfix.json --timerange 20250820- -t 5m 15m 1h 4h
   ```
4. **检查运行中的 bot**：`docker ps` 查看是否有策略正在 dry-run 或 live。**正在运行的策略文件不要修改**（`user_data/strategies/` 是 Docker volume 挂载的）。
5. **确认就绪**：向人类确认 setup 完成，然后开始实验循环。

## 执行环境

### 本地（开发 + 回测）

- **Freqtrade 运行在 Docker 中**，所有 freqtrade 命令通过 `docker exec freqtrade freqtrade ...` 执行
- **本地没有安装 freqtrade CLI**
- 策略文件在 `user_data/strategies/`（已 gitignore，提交时用 `git add -f`）
- 配置文件在 `user_data/`：`config.json`（现货）、`config_nfix.json`（合约+代理）、`config_mlpulse.json`（FreqAI）
- 数据：Binance 合约，10 个主流币对，5m/15m/1h/4h 时间周期

### 远程服务器（Dry Run + Prod）

- **服务器**：`ubuntu@13.212.151.85`（AWS 新加坡）
- **SSH**：`ssh -i ~/.ssh/yehui-ap-east.pem ubuntu@13.212.151.85`
- **部署目录**：
  - `/opt/freqtrade/dryrun/` → checkout `dryrun` 分支
  - `/opt/freqtrade/prod/` → checkout `prod` 分支
- **部署方式**：本地 push 到 origin → 远程 git pull → 重启容器
- 远程服务器直连 Binance（无需代理），使用 `config_remote.json`

## 实验循环

**你只修改策略文件**（`user_data/strategies/<StrategyName>.py`）和对应的配置文件。这是你的全部实验空间。

### LOOP FOREVER:

1. **选择实验方向**：从 `todo.md` 待办列表中选一个，或基于 `results.tsv` 分析提出新假设。
2. **创建/切换分支**：
   ```
   git checkout dev && git pull
   git checkout -b strategy/<StrategyName>  # 新策略
   # 或 git checkout strategy/<StrategyName>  # 继续迭代
   ```
3. **编写/修改策略代码**：直接修改策略文件。
4. **git commit**：提交当前代码（即使还没跑回测，先保存快照）。
   ```
   git add -f user_data/strategies/<StrategyName>.py
   git commit -m "experiment(strategy): <简述本次实验的假设>"
   ```
5. **运行回测**：
   ```
   docker exec freqtrade freqtrade backtesting \
     --strategy <StrategyName> \
     --config /freqtrade/user_data/<config>.json \
     --timerange 20250820- \
     > backtest.log 2>&1
   ```
   - 将输出重定向到 log 文件，**不要让大量输出灌进 context**
   - 结束后用 `tail -n 50 backtest.log` 读取结果摘要
6. **提取关键指标**：从回测输出中提取 Sharpe、总收益%、DD%、WR%、交易数。
7. **判断 keep/discard**：
   - **keep 标准**（全部满足才 keep）：
     - Sharpe Ratio > 1.5（理想 > 3.0）
     - Max Drawdown < 10%（理想 < 5%）
     - 总交易次数 > 30
     - Win Rate > 40%
     - 总收益 > 0%
   - 如果**达标** → **keep**：记录结果到 `results.tsv`，保留 commit
   - 如果**未达标** → **discard**：记录结果到 `results.tsv`（status=discard），`git reset --hard HEAD~1` 回退
8. **可选：Hyperopt 优化**（仅对 keep 的策略）：
   ```
   docker exec freqtrade freqtrade hyperopt \
     --strategy <StrategyName> \
     --config /freqtrade/user_data/<config>.json \
     --hyperopt-loss SharpeHyperOptLoss \
     --spaces roi stoploss trailing \
     --epochs 300 \
     --timerange 20250820- \
     > hyperopt.log 2>&1
   ```
   - 先优化 `--spaces roi stoploss trailing`，再优化 `--spaces buy sell`
   - 将最优参数写回策略文件，重新回测验证
   - 如果 hyperopt 后指标提升 → commit 并更新 `results.tsv`
   - 如果 hyperopt 后指标下降 → revert
9. **更新 `results.tsv` 和 `todo.md`**：
   - 追加本次实验结果到 `results.tsv`
   - 更新 `todo.md`：勾选完成项、添加新发现的待办、更新策略对比表
10. **回到步骤 1**，选择下一个实验方向。

### Timeout 处理

- 回测预期耗时：普通策略 < 5 分钟，FreqAI 策略 < 30 分钟
- 如果回测超过 30 分钟无输出，kill 并视为 crash
- Hyperopt 每轮预期 10-30 分钟（300 epochs），超过 60 分钟视为 crash

### Crash 处理

- 如果回测崩溃（import error、配置错误等），用 `tail -n 50 backtest.log` 读取报错
- 简单 bug（拼写错误、缺少 import）→ 修复后重跑
- 根本性问题（架构不兼容、数据缺失）→ 记录 crash 到 `results.tsv`，skip 并继续下一个实验
- 连续 3 次 crash 同一方向 → 放弃该方向，换一个实验

### **NEVER STOP**

实验循环开始后，**不要暂停问人类是否继续**。不要问"要不要继续？"。人类可能在睡觉，期望你持续工作到被手动中断。如果跑完了 `todo.md` 上所有待办，就自己想新的实验方向：

- 重读已有策略代码，寻找优化空间
- 组合之前接近成功的实验（如两个 Sharpe 接近达标的策略做特征融合）
- 尝试更激进的架构变化（不同时间周期、不同指标组合、不同 ML 模型）
- 搜索学术论文或量化社区的新想法

## 结果追踪（results.tsv）

所有实验结果记录在项目根目录 `results.tsv`（tab 分隔，不要用逗号）。

表头和格式：

```
commit	strategy	version	sharpe	profit_pct	dd_pct	wr_pct	trades	leverage	status	description
```

字段说明：
- `commit`：git commit hash（短 7 位）
- `strategy`：策略类名
- `version`：版本号（v1, v2, ...）
- `sharpe`：Sharpe Ratio（crash 时填 0.00）
- `profit_pct`：总收益百分比（crash 时填 0.00）
- `dd_pct`：最大回撤百分比（crash 时填 0.00）
- `wr_pct`：胜率百分比（crash 时填 0.00）
- `trades`：总交易次数（crash 时填 0）
- `leverage`：杠杆倍数
- `status`：`keep`、`discard`、`crash`、`hyperopt`、`dry-run`、`live`
- `description`：简短描述本次实验内容

**注意**：`results.tsv` 不要 git commit，保持 untracked。

## 策略上线流水线

```
[回测通过] → [checkout 到 dryrun 分支] → [部署到远程] → [Dry Run 7天] → [checkout 到 prod 分支] → [部署到远程] → [Live]
```

### Dry Run 上线标准（全部满足）

- Sharpe > 2.0
- Max DD < 5%
- 交易次数 > 50
- Win Rate > 45%
- hyperopt 后的参数已固化到策略文件或 JSON

### Dry Run → Live 上线标准（全部满足）

- Dry Run 运行 >= 7 天
- 模拟盈利为正
- 实际表现与回测偏差 < 30%
- 无单笔亏损超过总资金 5% 的交易

### 部署操作

**添加策略到 dryrun**：
```bash
# 1. 本地：checkout 策略文件到 dryrun 分支
git checkout dryrun
git checkout strategy/XX -- user_data/strategies/XX.py user_data/strategies/XX.json
git commit -m "chore(dryrun): add XX for dry-run validation"
git push origin dryrun

# 2. 远程：拉取并启动
ssh -i ~/.ssh/yehui-ap-east.pem ubuntu@13.212.151.85 \
  "cd /opt/freqtrade/dryrun && git pull && python3 scripts/multi_dryrun.py up"
```

**提升策略到 prod**：
```bash
# 1. 本地：checkout 策略文件到 prod 分支
git checkout prod
git checkout dryrun -- user_data/strategies/XX.py user_data/strategies/XX.json
git commit -m "chore(prod): promote XX after dry-run validation"
git push origin prod

# 2. 远程：拉取并启动
ssh -i ~/.ssh/yehui-ap-east.pem ubuntu@13.212.151.85 \
  "cd /opt/freqtrade/prod && git pull && python3 scripts/multi_dryrun.py up"
```

上线后在 `results.tsv` 中更新 status 为 `dry-run` 或 `live`。

## 约束边界

### 你 CAN 做的：

- 修改 `user_data/strategies/` 下的策略文件（创建新的或修改已有的）
- 创建/修改 `user_data/` 下的配置文件
- 运行回测、hyperopt、下载数据
- 自主使用 Git 管理版本（创建分支、提交、推送）
- 通过 SSH 部署到远程服务器（pull + restart）
- 修改 `todo.md` 和 `results.tsv`

### 你 CANNOT 做的：

- 交易 Meme 币、小市值山寨币或非加密货币资产
- 使用超过 20x 的杠杆
- 跳过回测直接上线 Dry Run 或 Live
- 修改 freqtrade 框架源码（`freqtrade/` 目录下的核心代码）
- 将 API Key 硬编码到任何文件中
- 删除或覆盖正在 Dry Run / Live 运行中的策略文件

## 交易范围

只允许交易主流加密货币合约（市值前 20）：

BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, DOT, LINK 等

## Git 规范

### 分支模型

```
develop              <- upstream freqtrade（只同步上游，不开发）
  └── dev            <- 通用基础（脚本、配置、.claude/）。不存策略文件
        ├── strategy/XX  <- 从 dev 分叉，单个策略开发+回测
        ├── dryrun       <- 从 dev 分叉，只放正在 dry-run 验证的策略
        └── prod         <- 从 dev 分叉，只放实盘运行的策略
```

- **dev**：只有通用工具（scripts/、ops/、.claude/、configs），**不存策略文件**
- **strategy/XX**：从 dev 分叉，包含单个策略的 .py 和 .json 文件
- **dryrun**：从 dev 分叉，包含所有正在 dry-run 的策略文件。远程部署到 `/opt/freqtrade/dryrun/`
- **prod**：从 dev 分叉，包含所有实盘运行的策略文件。远程部署到 `/opt/freqtrade/prod/`

### 策略上线流程

```
strategy/XX -> (回测通过) -> dryrun -> (7天验证通过) -> prod
```

**strategy/XX -> dryrun**（回测通过，开始 dry-run）：
```bash
git checkout dryrun
git checkout strategy/XX -- user_data/strategies/XX.py user_data/strategies/XX.json
git commit -m "chore(dryrun): add XX for dry-run validation"
git push origin dryrun
```

**dryrun -> prod**（dry-run 验证通过，上实盘）：
```bash
git checkout prod
git checkout dryrun -- user_data/strategies/XX.py user_data/strategies/XX.json
git commit -m "chore(prod): promote XX after dry-run validation"
git push origin prod
```

**移除策略**：
```bash
git checkout dryrun  # 或 prod
git rm user_data/strategies/XX.py user_data/strategies/XX.json
git commit -m "chore(dryrun): remove XX - <原因>"
```

### Commit 规范

```
feat(strategy): add <StrategyName> - <核心思路简述>
optimize(strategy): <StrategyName> v<N> hyperopt - Sharpe X.XX, +Y.YY%, DD Z.ZZ%
experiment(strategy): <StrategyName> - <本次实验假设>
fix(strategy): <StrategyName> - <修复内容>
chore(dryrun): add/remove <StrategyName> - dry-run 管理
chore(prod): promote/remove <StrategyName> - 实盘管理
```

### 合并规则

- `strategy/XX` 分支**不合并到 dev**，只通过 checkout 文件的方式提升到 dryrun/prod
- 回测通过 -> 推送 strategy/XX 分支到 origin 备份，打 tag `v<N>-<StrategyName>`
- dev 只接受通用工具脚本的提交

## 关键经验（历史教训）

这些是之前实验积累的重要规律，在设计新策略时**必须**参考：

- **高杠杆陷阱**：20x 杠杆下 stoploss 是 leveraged return（-0.01 = 0.05% 价格变动即触发）。需要 stoploss=-0.15~-0.34 才合理
- **手续费侵蚀**：20x 杠杆的 round-trip fee = 0.05% x 20 x 2 = 2%，极大侵蚀利润
- **时间周期**：1h 比 5m/15m 噪声少很多，策略更稳定
- **Hyperopt 分步做**：先 `roi stoploss trailing`，再 `buy sell`，效果优于一次全做
- **BTC 趋势过滤**：在 ML 策略中加入 BTC 1h EMA50/200 + RSI>40 趋势过滤对熊市表现至关重要
- **宽止盈窄止损**：让赢家跑（宽 ROI），不要过度优化止损
- **FreqAI 注意**：
  - 回测 timerange 必须在足够的训练数据之后开始（不是数据起始点）
  - 需要 `--freqaimodel LightGBMClassifier` 标志
  - Docker 中需额外安装：`pip install datasieve lightgbm xgboost tensorboard`
  - 分类器概率列以类标签命名（如 "up"、"down"、"neutral"）

## todo.md 格式

`todo.md` 是跨 session 的工作记忆。格式如下：

```markdown
## 待办
- [ ] 具体可执行的实验描述

## 进行中
- [ ] 当前正在做的实验（开始日期：YYYY-MM-DD）

## 已完成
- [x] 完成的实验 -> 关键结果数据

## 策略版本对比
（从 results.tsv 生成的汇总表）

## 关键教训
- 本轮实验中发现的重要规律
```

每次实验完成后**必须**更新 `todo.md`。
