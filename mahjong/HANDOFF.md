# 支持并发训练的红中麻将环境

## Summary

在现有环境方案上增加进程安全、可序列化和高吞吐设计，使环境可直接用于 Stable-Baselines3 `SubprocVecEnv`，也可作为 Ape-X actor 进程中的本地环境批量产生 transition。默认训练模式只生成向量观测；视觉 RGB 观测仍可按配置启用，避免并发训练时无谓的渲染开销。

## 并发接口与生命周期

- `MahjongEnv` 保持标准 Gymnasium `Env`：
  - `reset(seed=..., options=...) -> (obs, info)`
  - `step(action) -> (obs, reward, terminated, truncated, info)`
  - `close()`
- 环境对象只保存自身牌局状态、配置和独立 RNG，不使用全局随机数、全局 pygame 状态或跨实例缓存。
- 增加可序列化的 `EnvConfig`，其中包含规则、奖励、观测模式、图像尺寸、机器人参数和 seed。
- 增加顶层工厂：
  - `make_env(config=None, seed=None, rank=0)`
  - `make_env_fn(config, seed, rank)`，返回可被 multiprocessing pickle 的零参数闭包。
- 每个并发实例使用 `base_seed + rank` 初始化，并在 `reset(seed=...)` 时重新派生局内 RNG，保证不同 worker 独立且可复现。
- 默认 multiprocessing 使用 `spawn` 或 `forkserver` 兼容路径；worker 创建时不初始化 pygame，只有显式 `render_mode="human"` 时才创建窗口。

## 观测与吞吐模式

- `observation_mode` 支持：
  - `"vector"`：默认，高吞吐训练模式；
  - `"rgb"`：只返回图像；
  - `"both"`：同时返回向量和图像，用于视觉实验。
- `render_mode="rgb_array"` 与环境观测解耦；需要采集图像时才调用离屏渲染。
- `SubprocVecEnv` 使用固定 shape、固定 dtype 的 Gymnasium spaces，避免 worker 间动态空间。
- `info` 中提供：
  - `action_mask`；
  - `episode_id`、`step_index`；
  - `terminal_reason`；
  - `win`、`draw`、`robbed_kong`；
  - 可选的潜力值和牌局统计。
- `action_mask` 同时作为 observation 字段或 `info` 字段提供，并保持固定长度 141，方便策略网络和 actor 直接过滤非法动作；自摸由环境自动终止，不再占用动作编号。

## Ape-X / actor 兼容

- 增加 `transition.py` 定义稳定的 transition schema：
  - `obs`、`action`、`reward`、`next_obs`；
  - `terminated`、`truncated`；
  - `action_mask`、`next_action_mask`；
  - `episode_id`、`step_index`、`env_rank`。
- transition 使用 NumPy 原生 dtype 和可 pickle 数据结构，避免携带 pygame 对象、环境对象或不可序列化句柄。
- 增加 actor-side rollout 工具：
  - 一个进程内可运行多个环境；
  - 按固定步数或 episode 边界批量输出 transition；
  - 支持 n-step return 所需的局部缓存；
  - 支持将 RGB 字段关闭，减少进程间复制。
- 提供批量 `stack_transitions()` 与校验函数，保证传给 replay buffer 的数组 shape 一致。
- Ape-X 的优先级计算、网络参数同步和 replay server 保持在环境外；环境只负责高效、无状态污染地生成合法 transition。

## 代码组织

- `MahjongEnv/env.py`：单环境 Gymnasium 实现。
- `MahjongEnv/config.py`：可 pickle 的配置和默认参数。
- `MahjongEnv/vector.py`：`make_env_fn`、批量 rollout 和并发辅助函数。
- `MahjongEnv/transition.py`：Ape-X transition schema、批处理和校验。
- `MahjongEnv/renderer.py`：仅在显式 RGB/GUI 模式中初始化 pygame。
- `play.py`：唯一的人类 GUI 入口；GUI 进程与训练 worker 完全分离。
- `MahjongEnv/tests/`：规则、Gymnasium checker、并发和 transition 测试。

## 并发验收测试

- 使用 Gymnasium checker 检查环境接口和 spaces。
- 用 `SubprocVecEnv` 启动至少 4 个 worker，连续运行多个 episode，确认：
  - 无 pickle 错误；
  - 无 pygame 初始化错误；
  - 每个 worker 的 seed 和牌局序列独立；
  - batch observation、reward、done 和 action mask shape 正确。
- 使用 `spawn` 启动方式运行向量环境，覆盖 Linux 和无显示环境。
- 同一 seed 下单进程与多进程环境分别验证局内可复现性。
- 运行 Ape-X rollout 示例，确认 transition 可 pickle、批量堆叠、n-step 缓存边界和终局标志正确。
- 测量 vector-only 模式的每秒 step 数，并确保启用 RGB 后只是按配置增加成本。

## 默认值与兼容约束

- 默认 `observation_mode="vector"`，`render_mode=None`，适合 SubprocVecEnv 和 Ape-X。
- GUI 使用根目录的 `play.py` 进程，默认 `"both"` 或 `"rgb_array"`，不与训练 worker 共用 pygame。

GUI 支持点击手牌弃牌，以及碰、明杠、暗杠、加杠、胡、过等动作按钮；牌面使用中文名称，
每张手牌下方显示弃牌快捷键，右侧显示可用操作。三位牌友的所有弃牌和副露会以牌块形式显示；牌友仍使用环境中的随机出牌逻辑。
项目随附 `assets/fonts/NotoSansCJK-Regular.ttc`，渲染器会优先使用它，保证中文、数字和字母使用同一套完整字体。
本轮新摸的牌单独放在手牌最右侧并使用红框标记；出牌后剩余手牌重新排序，红中始终置于最左侧。
GUI 模式下每位牌友出牌后默认停留 1.5 秒并刷新桌面，可通过 `--opponent-delay` 调整；headless 模式不等待。
牌友停留期间会持续处理窗口事件并清空旧键盘输入，连续敲击空格不会在下一响应阶段重复触发“过”。
`play.py` 默认持续开始下一局；使用 `--episodes N` 可限定局数，首屏字体和 pygame 初始化采用缓存/按需初始化以缩短启动时间。
GUI 牌局结束后会显示终局结果（自摸胜利、牌友胡牌、牌友天胡或流局），等待 Enter/Space 确认后才开始下一局；headless 模式继续自动 reset。
默认 GUI 画布为 `1200×900`，牌友历史、我方弃牌和我方手牌分区之间有明确间隔。
结构化向量观测为 306 维：我方手牌 28、我方副露 28、四家弃牌 112、四家公开副露牌种 112、四家副露类型 16、状态标量/阶段标志/塑形特征 10。每个区块固定位置；对手暗杠只记录发生过暗杠，不泄露具体牌种。
湖南红中麻将只开放自摸胡：牌友弃牌进入响应阶段时不生成胡牌动作，只能碰、明杠或过；我方摸牌后才显示 `自摸 [W]`。
胡牌判定支持红中补在顺子开头的情况，例如 `红中+8条+9条` 可组成 `7/8/9条`。
每局从四位玩家中随机选择起手玩家；该玩家拿 14 张并先出牌。起手玩家若 14 张直接成胡，则标记为天胡，牌友天胡的牌局直接结束，我方天胡显示 `天胡 [W]`。
牌友弃牌使用启发式：红中永不弃出，优先弃孤张和不连接成顺/刻的牌；牌友摸牌后先判自摸，再决定弃牌。牌友碰牌策略暂不实现。
牌友出牌后会进入 `claim` 阶段，等待人类响应；人类选择过后才继续下一位牌友。每局专家数据保存为输出目录下
按北京时间命名的独立 NPZ 文件，历史文件不会被读取或覆盖。
- 环境不创建线程、不依赖进程共享内存；数据落盘由 actor 或训练器显式负责。
- 保留前述牌局规则、141 动作编码、红中万能牌、稠密奖励和 NPZ 专家数据格式。

## 人类专家数据保存

运行 `python play.py --episodes 2 --output expert_data`，每局结束立即在输出目录
生成一个独立的 `episode_YYYYMMDD_HHMMSS_ffffff.npz` 文件，时间固定采用 `Asia/Shanghai`，
精确到微秒。重复启动不会读取或重写历史数据，只缓存当前局的数据。随机牌友未改动。
文件完整写入并同步后才发布；文件名冲突时拒绝覆盖。中途退出的未完成局不保存，
之前已完成的局保留。

旧的 `--output expert_data.npz` 参数会将新局写入同名无后缀目录 `expert_data/`，
原有 NPZ/JSON 不动。每局 NPZ 保留原有 transition 数组字段，额外的 `metadata`
字段是 JSON 字符串（通过 `json.loads(data['metadata'].item())` 读取），不再单独写 JSON。
训练时按需加载目录中的各局文件；旧的合并数据集需要另行纳入训练输入。
