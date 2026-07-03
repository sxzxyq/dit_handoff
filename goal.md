请在当前服务器上从零构建一个新的、干净的 handoff / DIT 工作空间。

工作空间根目录固定为：

/home/qsh/dit

数据、大文件、训练输出、评估输出固定保存在：

/data/shared_folder/datasets/dit

旧仓库地址：

https://github.com/sxzxyq/dykj_lbm

使用模型官方地址：

https://huggingface.co/docs/lerobot/en/multi_task_dit

注意：
- 旧仓库不在本地。
- 只能通过 GitHub 查看旧仓库代码。
- 不要假设旧服务器路径存在。
- 不要写任何依赖 `/home/ubuntu/...`、旧 experiments 目录、旧 raw demos、旧 checkpoint 的命令。
- 不存在删除旧仓库、迁移旧数据、读取旧数据的问题。
- 旧 GitHub 仓库只作为只读参考，用来理解任务环境、task registration、相机配置、action space、旧脚本结构和旧失败点。
- 如果需要临时 clone 旧仓库作为 reference，只能放在 `/home/qsh/dit/_reference/`，并加入 `.gitignore`。
- 新工作区最终必须可以独立说明如何 setup、collect、validate、convert、train、open-loop eval、closed-loop eval。

本次目标：
1. 在 `/home/qsh/dit` 新建一个清晰、最小、可复现的新工作区。
2. 任务场景保持不变：仍然是 IsaacLab 双臂物体搬运 / handoff 任务。
3. 本次只实现一条最小闭环管线：

   通用 raw 数据采集 schema
   -> raw validator
   -> LeRobot dataset conversion
   -> 3 cameras + state26 -> absjoint18
   -> 官方 LeRobot MultiTask DiT 训练
   -> open-loop eval
   -> closed-loop eval

4. 本次不要实现其他输入输出版本。
5. 目录结构按功能分文件夹即可，不需要一开始精确到每个脚本。
6. 数据采集脚本必须是通用 raw collector，不要为 state26 或 absjoint18 特殊化。
7. converter 才负责定义具体训练 contract。本次只需要实现 3 cameras + state26 -> absjoint18 的 converter。
8. 本地尚未安装 LeRobot。可以自行新建 conda 环境并安装 LeRobot，尽量以官方 LeRobot MultiTask DiT 实现为准。
9. 训练脚本优先封装官方 `lerobot-train` 或当前官方等价入口，不要重写一套脱离官方实现的复杂训练主循环。
10. 所有 raw images、LeRobot dataset、checkpoint、video、report、cache 都必须写到 `/data/shared_folder/datasets/dit`，不要写进 `/home/qsh/dit`。

核心原则：
- raw 数据尽量保存环境中能直接读到的信息。
- 采集脚本不要手工推断子任务、active arm、phase、first close、TCP-cube distance、relative pose、progress 等派生标签。
- 如果环境 observation 或 info 里已经直接提供某个字段，可以原样保存，并标注为 env-provided。
- 如果需要后续诊断用派生指标，应在 validator/eval/report 阶段计算，不要混进 raw ground-truth schema。
- raw 数据要明确区分：
  - directly observed from env
  - action commanded by collector
  - env returned info
  - later derived diagnostics
- converter 可以从 raw 中选择字段组成 state26，但不要在采集阶段为某个 converter 改 raw 格式。
- 不要写一个巨型万能 train/eval 脚本。
- 本次只做 state26 -> absjoint18；不要提前实现其他 contract，也不要创建一堆未来版本占位脚本。

非常重要的 18D action 约束：
- 本次要再试 18D absolute joint target。
- 但是不要再默认把 IK 执行后的 `post joint state` 当作 18D action label。
- 18D absjoint 的训练 label 必须来自实际发送给 Joint-Pos action space 的 commanded joint target。
- 如果专家内部使用 IK、解析方法或其他方法计算目标关节角，这是允许的；但必须：
  1. 把这个 18D joint target 作为 env action 发送；
  2. 在 raw 数据中保存为 commanded action；
  3. 同时保存 env.step 后观测到的 post joint state；
  4. converter 默认使用 commanded action，不使用 post observed joint state。
- 禁止把 `env.step(action)` 之后观测到的 post joint_pos 当成默认 action label。
- post joint_pos 只能作为 post observation / tracking diagnostic。
- 如果为了兼容旧问题需要写 legacy diagnostic，必须显式标记为 legacy，不得进入默认 state26 -> absjoint18 pipeline。

推荐目录结构只需要到子文件夹层级：

/home/qsh/dit/
  README.md
  .gitignore
  docs/
  configs/
    setup/
    collect/
    convert/
    train/
    eval/
  src/
    dit_handoff/
      env/
      collect/
      data/
      convert/
      train/
      eval/
      lerobot_bridge/
      utils/
  scripts/
    setup/
    collect/
    validate/
    convert/
    train/
    eval/
  tests/

大文件目录固定为：

/data/shared_folder/datasets/dit/
  raw/
  lerobot/
  runs/
  checkpoints/
  eval/
  reports/
  videos/
  cache/

要求：
- `/home/qsh/dit/.gitignore` 必须忽略：
  - `_reference/`
  - `__pycache__/`
  - `.pytest_cache/`
  - `.mypy_cache/`
  - `.ruff_cache/`
  - wandb/
  - tensorboard/
  - outputs/
  - logs/
  - *.pt
  - *.pth
  - *.safetensors
  - *.mp4
  - *.png generated outputs
  - any symlinked dataset/cache directories
- 不要把 raw images、dataset、checkpoint、video 放进 git-tracked workspace。
- 如果需要在 workspace 里访问数据，可以创建软链接，但真实数据必须在 `/data/shared_folder/datasets/dit`。

环境安装要求：
- 本地尚未安装 LeRobot。
- 创建 conda 环境安装脚本，默认环境名可以是 `dit_lerobot`，但允许通过环境变量覆盖。
- 优先使用官方 LeRobot 安装方式，例如：

  pip install "lerobot[multi_task_dit]"

  但需要以当前官方文档和实际可安装版本为准。
- 需要主动查看当前官方 LeRobot MultiTask DiT 文档和当前安装版本 API。如果官方 CLI 名称、参数名或 dataset API 已变化，以当前可安装版本为准，并在文档里记录差异。
- 写一个 setup/check 脚本，至少检查：
  - conda env 是否存在
  - python version
  - torch 是否可用
  - cuda 是否可用
  - lerobot 是否可 import
  - MultiTask DiT policy / trainer 入口是否可用
  - IsaacLab 是否可用
  - handoff task 是否能注册
  - headless env 是否能 reset
- IsaacLab 可能有自己的 Python/launcher。如果 collect 和 closed-loop eval 必须通过 IsaacLab launcher 运行，请不要强行绕过；应在文档中说明：
  - 训练使用哪个 conda env
  - 采集/eval 使用哪个 Python/launcher
  - LeRobot policy 在 eval 时如何加载
  - 这两个 Python 环境之间如何共享 checkpoint/dataset/config
- 尽量不要手写一套脱离官方 LeRobot 的训练 loop。
- 训练脚本应优先作为官方 `lerobot-train` 或当前官方等价入口的 thin wrapper。
- 如果当前 LeRobot 版本官方入口不可用，才允许写最小 fallback，并必须在 README 和 docs/setup_notes.md 解释原因。

旧仓库参考方式：
- 可以通过 GitHub 查看旧仓库。
- 参考内容包括：
  - IsaacLab handoff task cfg
  - task registration
  - camera names 和 camera order
  - robot naming convention
  - Joint-Pos action space
  - 旧 pipeline 的失败点
- 不要假设旧仓库在本地。
- 不要依赖旧数据。
- 不要复用旧脚本的大量分支结构。
- 不要直接复制旧脚本。
- 新工作区不能把 `_reference` 目录作为运行时依赖。

任务环境要求：
- 场景保持旧 handoff 任务不变。
- 双臂、物体、目标区、三路相机保持与旧任务一致。
- 本次 18D absjoint closed-loop eval 使用 Joint-Pos action interface。
- 如果旧仓库已有 Joint-Pos handoff task，则复用其配置思想和 task id。
- 不要重写成另一个任务。
- 不要改变相机语义、机器人左右臂语义、目标区语义。
- 三路相机语义固定为：
  - wrist_rgb
  - observer_wrist_rgb
  - global_rgb
- 左右臂命名、joint order、action order 必须写入 manifest 和 docs。
- quaternion convention 必须明确，例如 wxyz 或 xyzw，不允许静默混用，旋转优先使用轴角。
- gripper_opening 的定义必须明确，例如 finger joint sum、单侧 opening，或 env 直接 term。

raw 数据采集设计：
- 写一个通用 raw collector。
- collector 不应该叫 state26，也不应该只服务 absjoint18。
- collector 的职责是：
  - 创建 handoff env
  - 运行 expert / scripted policy
  - 保存每一步 env 可直接读取的信息
  - 保存实际发送给 env 的 action
  - 保存三路图像
  - 保存 env 返回的 reward/terminated/truncated/info
  - 保存 episode-level metadata
- 本次默认 action mode 是 Joint-Pos 18D commanded target，因为后续要做 state26 -> absjoint18。
- 但 collector 的 raw schema 要通用，后续可以通过 converter 做不同输入输出，而不是改采集格式。

raw dataset 目录建议：

/data/shared_folder/datasets/dit/raw/<dataset_name>/
  dataset_manifest.json
  episodes/
    episode_000000/
      episode_meta.json
      steps.parquet 或 steps.jsonl
      images/
        wrist_rgb/
        observer_wrist_rgb/
        global_rgb/
      optional_video.mp4
  validation_report.json

raw 每个 episode 至少保存：
- schema_version
- dataset_name
- task_name
- env_id
- action_interface
- fps / sim dt / decimation，如果环境或配置可直接得到
- camera_names
- image_size
- joint_names
- action_names
- seed
- randomization_enabled
- randomization_config
- collection_command
- code version / git commit，如果可得
- success definition，如果环境中有明确 success info
- episode length
- episode success，如果环境直接提供或 collector 有明确最终判定

raw 每个 step 至少保存：

基础字段：
- episode_id
- step_index
- sim_time，如果环境直接可得
- wall_time，可选
- reward
- terminated
- truncated
- env_info 原样可序列化部分

pre observation：
- left/right joint_pos，如果 env 可直接读到
- left/right joint_vel，如果 env 可直接读到
- left/right tcp_pos_w，如果 env 可直接读到
- left/right tcp_quat_w，如果 env 可直接读到
- left/right gripper opening 或 finger joint，如果 env 可直接读到
- cube pose，如果 env 可直接读到
- target pose，如果 env 可直接读到
- environment observation dict 中其他直接存在的数值项
- 三路 image path

action：
- raw_env_action
- action_interface
- action_layout
- joint_target_18_commanded，如果当前 action mode 是 Joint-Pos 18D
- 夹爪 command，如果它是 action 的一部分或可直接记录
- 不要用 post joint_pos 填充 joint_target_18_commanded

post observation：
- 与 pre observation 同类字段
- post joint_pos 必须保存，但只能是 post observation
- post tcp pose / gripper / cube pose 等如果可直接读到，也保存

不要在 raw step 中保存以下派生字段，除非 env 原生 info 已经直接返回：
- subtask
- phase
- active_arm
- TCP-cube distance
- first close flag
- grasp candidate
- area success heuristic
- manually computed relative pose
- manually computed progress
- manually computed red/yellow stage

关于末端相对姿态：
- 如果任务环境 observation manager 已经直接提供 relative pose term，则原样保存，字段名标记为 env-provided。
- 如果环境没有直接提供，不要在 collector 中手工计算。
- 后续如果需要相对位姿用于训练或诊断，在 converter 或 analysis/eval 脚本中从 raw 的直接观测字段计算，并在输出 manifest 中标记为 derived。
- 本次 converter 只使用 state26，不使用 relative pose。

随机化：
- 在 collector/config 中保留 randomization 开关接口。
- 至少支持：
  - enable_randomization true/false
  - seed
  - randomization_profile
  - cube pose randomization 参数，如果任务已有
  - 相机图像增强随机化
  - 其他随机化参数可以先留接口
- 如果某项随机化尚未实现，manifest 必须明确写 `not_implemented` 或 `disabled`。
- 不允许把 dataset 命名为 full randomization，但实际没有记录随机化内容。
- 本次不要求一定完整实现所有视觉/相机/材质随机化，但接口和 manifest 必须清楚。

raw validator：
实现 raw dataset validator，检查：
- dataset_manifest.json 是否存在
- episode_meta.json 是否存在
- steps 文件是否存在
- 三路图像路径是否存在
- step_index 是否连续
- timestamp/sim_time 是否单调，如果存在
- pre/post observation 字段是否完整
- action 维度是否等于 manifest 中 action layout
- 当 action_interface 是 Joint-Pos 18D 时，`joint_target_18_commanded` 必须存在
- `joint_target_18_commanded` 不得是 post joint_pos 回填出来的字段；如果检测到 action label 来自 post joint_pos，应报错或标记 invalid
- joint_names / action_names 顺序一致
- train/val split 必须按 episode，不按 frame 混
- 输出 validation_report.json

LeRobot conversion：
本次只实现一个 converter：
- raw -> LeRobot
- contract: 3 cameras + state26 -> absjoint18

observation.state 26D 定义固定为：
- left joint_pos 9D
- left tcp_pos_w 3D
- left gripper_opening 或可等价表示的 gripper scalar 1D
- right joint_pos 9D
- right tcp_pos_w 3D
- right gripper_opening 或可等价表示的 gripper scalar 1D

action 18D 定义固定为：
- left joint target 9D
- right joint target 9D

converter 要求：
- action 必须来自 raw 中的 `joint_target_18_commanded`
- images 固定为三路：
  - wrist_rgb
  - observer_wrist_rgb
  - global_rgb
- 写出 LeRobot dataset
- 写 conversion_summary.json
- 写 manifest.json
- 写 feature names
- 写 state layout
- 写 action layout
- 写 image feature names
- 写 train/val split
- 写 stats source
- 写 raw dataset path
- 不要混入训练逻辑
- 不要用参数切换成别的输入输出 contract
- 不要提前实现其他 converter
- 尽量输出当前官方 LeRobotDataset v3 或当前安装版本支持的官方格式
- low-dimensional state/action/timestamp 按官方格式保存
- 多相机图像/视频按官方格式保存
- meta/info.json、meta/stats.json、meta/tasks.jsonl、meta/episodes 等按当前官方 API 生成
- 如果使用 LeRobotDataset.create/add_frame/save_episode 这类 API，最后必须调用 finalize 或当前版本等价收尾方法，确保 parquet/video writer 正常关闭，dataset 可以被官方 loader 读取

MultiTask DiT 训练参数设计：
- 官方推荐的 horizon / n_action_steps 本质是按时间窗口，而不是固定步数。
- 训练配置必须从 raw manifest 或 LeRobot dataset meta 中读取 fps。
- 本项目必须在 raw manifest、LeRobot dataset manifest、training config、checkpoint config 中显式记录：
  - control_hz / fps
  - horizon_steps
  - horizon_seconds
  - n_action_steps
  - n_action_seconds
  - n_obs_steps
  - action_dim
  - action representation

默认 official-time profile：
- horizon_seconds = 1.0
- n_action_seconds = 0.8
- horizon = round(fps * horizon_seconds)
- n_action_steps = round(fps * n_action_seconds)

参考：
- 如果 fps=30，则接近：
  - horizon ≈ 32 或 30
  - n_action_steps ≈ 24
- 如果 fps=50，则接近：
  - horizon ≈ 50
  - n_action_steps ≈ 40

本次 state26 -> absjoint18 的正式训练默认采用 official-time profile：
- n_obs_steps = 2
- horizon_seconds ≈ 1.0
- n_action_seconds ≈ 0.8
- objective = diffusion
- noise_scheduler_type = DDPM
- num_train_timesteps = 100，如果当前 LeRobot 版本支持
- num_inference_steps = 10，如果当前 LeRobot 版本支持
- 不要把 n_action_steps=8 写死成正式默认值

允许准备短 action chunk 的诊断配置，但不要当作官方默认：
- n_action_steps = 8
- n_action_steps = 16

这些短 chunk 配置只用于定位接触阶段闭环恢复问题，例如抓取、闭爪、接触漂移，不作为第一版正式 baseline 的默认参数。

关于旧失败经验：
- 旧 h50/a40 失败不应被解释为官方 0.8 秒推荐一定错误。
- 旧实验同时存在 18D target/interface mismatch，即 action label 来自 IK 执行后的 post joint state，而不是 Joint-Pos commanded target。
- 新版必须先确保 18D action label 是 commanded Joint-Pos target，再比较 official-time profile 和短 chunk 诊断 profile。

训练：
- 本次只实现 state26 -> absjoint18 的训练入口。
- 使用官方 LeRobot MultiTask DiT 训练方式。
- 训练脚本是 thin wrapper，不要复制旧仓库里的自定义复杂训练主循环。
- 如果官方入口是 `lerobot-train`，优先使用它。
- 如果当前安装版本入口不同，使用当前官方入口并写入 README。
- policy.type 必须是 multi_task_dit。
- 训练前必须先跑：
  - raw validator
  - converter validator
  - LeRobot dataset smoke check
- 训练配置必须显式记录：
  - dataset path 或 repo id
  - policy type = multi_task_dit
  - horizon
  - n_obs_steps
  - n_action_steps
  - horizon_seconds
  - n_action_seconds
  - image features
  - state dim = 26
  - action dim = 18
  - batch size
  - gradient accumulation
  - effective batch size
  - learning rate
  - output dir
  - seed
  - objective
  - diffusion / inference steps
- 输出目录必须在：

  /data/shared_folder/datasets/dit/runs/

- checkpoint 必须在：

  /data/shared_folder/datasets/dit/checkpoints/

- 支持 smoke training，例如 5 或 10 steps，用于确认 pipeline 可运行。
- smoke training 只能用于验证 pipeline，不代表模型质量。

官方 MultiTask DiT 训练要点：
- Batch size：
  - 官方推荐 effective batch size 尽量达到 192-320。
  - 当前 GPU 如果不够，使用 gradient accumulation。
  - 配置中区分：
    - physical_batch_size
    - gradient_accumulation_steps
    - effective_batch_size = physical_batch_size * gradient_accumulation_steps
  - 如果 effective batch size 低于 64，README 和 train log 中必须标记可能训练不稳定。
- Training steps：
  - 正式训练建议至少 30000 steps。
  - 如果闭环出现 idling/no motion，而数据量已经足够，可允许继续到 50000-100000 steps，并记录 resume 来源。
- Objective：
  - 第一版默认 objective=diffusion。
  - flow_matching 只作为可选实验开关；当 diffusion 生成质量差、动作不平滑或不一致时再测试，不要作为第一版默认。
- Model architecture：
  - 中等数据规模默认使用官方中等配置，如果当前官方版本支持：
    - num_layers=6
    - hidden_dim=512
    - num_heads=8
    - use_rope=true
  - 小数据集可以降到 4 layers。
  - 大数据集可以升到 8 layers。
  - 不要第一版随意扩大模型；先保证数据/action contract 正确。
- Vision encoder and images：
  - 默认三路相机输入：
    - wrist_rgb
    - observer_wrist_rgb
    - global_rgb
  - image_crop_shape 默认 224x224，除非官方当前推荐不同。
  - 随机 crop/augmentation 只用于训练。
  - open-loop eval 和 closed-loop eval 必须使用确定性 preprocessing。
  - 如果使用 resize/crop，训练、open-loop eval、closed-loop eval 必须完全一致。
  - separate_rgb_encoder_per_camera 可以保留为开关，但默认关闭，除非显存足够；开启时要在 config 中记录 VRAM 风险。
  - 图像增强只应在训练加载 dataset 时做，不要污染 raw 数据。
- Learning rate：
  - 默认 optimizer_lr=2e-5，除非当前官方推荐不同。
  - vision_encoder_lr_multiplier=0.1，除非当前官方推荐不同。
  - 如果训练 loss 不稳定或发散，允许在 1e-5 到 3e-4 范围内调学习率，但每次实验必须记录。
  - checkpoint、eval、log frequency 都要显式写进 config，不要依赖隐式默认值。
- Language instruction：
  - MultiTask DiT 使用 text conditioning。
  - 即使当前只有单任务，也要在 raw manifest、LeRobot meta/tasks、train config、closed-loop eval 中使用一致的 language instruction。
  - 不要让 collector、converter、eval 各自写不同的语言字符串。
- Action representation：
  - 本次 state26 -> absjoint18 使用 absolute joint-space action。
  - 本项目必须保证：
    - action 是 Joint-Pos action interface 下实际 commanded 的 18D target
    - action 不是 env.step 后观测到的 post joint_pos
    - action_names 与 Joint-Pos env action order 完全一致
    - action 的 coordinate/control frame 在 manifest 中明确说明
  - relative/delta action 不在本次实现范围内，不要提前混入。
- Dataset transforms：
  - raw collector 保存原始图像，不做训练增强。
  - image transforms 只在训练 dataset loading 阶段启用。
  - transform 配置必须写入 train config。
  - open-loop eval 和 closed-loop eval 不使用随机增强，只使用确定性 preprocessing。

closed-loop eval 对 horizon/action steps 的要求：
- 默认从 checkpoint/config 读取 horizon 和 n_action_steps。
- 不允许 eval 脚本静默覆盖 n_action_steps。
- 如果用户显式覆盖，必须在终端 warning、summary.json、policy_calls.jsonl 里记录：
  - checkpoint_n_action_steps
  - override_n_action_steps
  - checkpoint_n_action_seconds
  - override_n_action_seconds
- 每次 policy call 都要记录 action chunk 的 shape 和实际执行了多少步。

open-loop eval：
- 本次只实现 state26 -> absjoint18 的 open-loop eval。
- 从 LeRobot val split 读取样本。
- 加载 checkpoint。
- 使用与训练一致的 preprocessing。
- 比较 predicted action 和 target action。
- 输出：
  - mse
  - mae
  - per-dim mae
  - action chunk 维度检查
  - summary.json
- 输出目录在：

  /data/shared_folder/datasets/dit/reports/

closed-loop eval：
- 本次只实现 state26 -> absjoint18 的 closed-loop eval。
- 使用 Joint-Pos handoff env。
- eval 脚本不要混 14D、relative、delta、legacy 分支。
- live observation adapter 只负责把 IsaacLab env 当前 observation 映射为：
  - 三路图像
  - state26
  - language，如果 policy/config 需要
- action adapter 只负责把 policy 输出的 18D absolute joint target 映射回 Joint-Pos env action。
- 保存：
  - summary.json
  - steps.jsonl
  - policy_calls.jsonl
  - videos，如果启用相机
- 记录每次 policy call 的 action chunk。
- 记录每步实际发送的 18D action。
- 记录 env 返回的 reward/terminated/truncated/info。
- 可以在 eval report 中计算诊断指标，例如 TCP-cube distance、first close step 等，但这些是 eval diagnostics，不是 raw schema 字段。
- closed-loop eval 输出目录在：

  /data/shared_folder/datasets/dit/eval/

failure debugging 要求：
- 如果 closed-loop idling/no motion：
  1. 先检查 action label 是否来自 commanded Joint-Pos target；
  2. 再检查 action_names / joint order 是否和 env action order 一致；
  3. 再检查 language instruction 是否进入 policy；
  4. 再检查 normalization stats 和 image preprocessing；
  5. 再考虑增加数据量或训练到 50k-100k。
- 如果动作生成质量差：
  1. 先保留 diffusion baseline；
  2. 再尝试 flow_matching。
- 如果推理太慢：
  1. 先减少 num_inference_steps 或使用 DDIM；
  2. 再考虑更小模型或 bfloat16/use_amp。

README 要包含：
1. 项目目标
2. 旧仓库只读 GitHub reference 的说明
3. 本次只实现 state26 -> absjoint18 最小闭环管线
4. 目录结构
5. 数据目录 `/data/shared_folder/datasets/dit`
6. conda/LeRobot/IsaacLab setup 命令
7. raw collect 命令
8. raw validate 命令
9. convert to LeRobot 命令
10. smoke train 命令
11. formal train 命令
12. open-loop eval 命令
13. closed-loop eval 命令
14. raw schema 说明链接
15. horizon / n_action_steps 如何按 fps 换算
16. 当前已知痛点

docs/ 至少包含：
- data_contract.md
- state26_absjoint18_contract.md
- setup_notes.md
- known_pain_points.md

known_pain_points.md 必须写清楚：
- 旧仓库不在本地，只能作为 GitHub reference。
- 新工作区不能依赖旧服务器路径或旧数据。
- 18D absolute joint target 必须来自 commanded Joint-Pos action，不能默认来自 post joint state。
- 如果 Joint-Pos commanded expert replay 不稳定，训练结果不能证明 policy 不行，首先说明 data/action contract 有问题。
- IsaacLab collect/eval 和 LeRobot training 可能存在 Python 环境/launcher 差异，需要明确记录。
- raw collector 只保存 env 直接可读字段，不保存手工推断 phase/active_arm/subtask。
- 随机化需要 manifest 记录开关和参数，未实现项必须明确标记。
- 三路图像 feature name 和顺序必须贯穿 collect/convert/train/eval。
- train/val 必须按 episode split。
- horizon / n_action_steps 应按 fps 换算，不要盲目固定步数。
- 长 action chunk 在接触任务中有闭环风险；短 chunk 只作为诊断配置，不作为官方默认。
- 大量图像、视频、checkpoint 必须写到 `/data/shared_folder/datasets/dit`，不要进 git workspace。
- quaternion convention 必须明确，例如 wxyz 或 xyzw，不允许静默混用。
- gripper_opening 的定义必须明确，是 finger joint sum、单侧 opening，还是 env 直接 term。
- LeRobot 官方版本可能变化，所有 CLI/API 差异必须在 setup_notes.md 记录。

测试与验收：
至少实现以下 smoke：
1. setup check：能 import lerobot、torch、IsaacLab，并确认 handoff task 可注册。
2. env smoke：headless reset 成功。
3. collect smoke：采集一个很短 episode 或固定步数 raw 数据，写入 `/data/shared_folder/datasets/dit/raw/`。
4. raw validate smoke：validator 通过。
5. convert smoke：生成 state26 -> absjoint18 LeRobot dataset。
6. train smoke：官方 LeRobot MultiTask DiT 跑 5 或 10 steps。
7. open-loop smoke：能在小 val subset 上输出 action MAE。
8. closed-loop smoke：能 headless 运行至少 10 env steps，并保存 policy_calls/steps summary。
9. python -m py_compile 通过。
10. bash -n 通过。
11. pytest 通过基础 schema/layout/action adapter 测试。

禁止事项：
- 不要把各种未来输入输出版本都写进本次实现。
- 不要提前实现其他 converter/train/eval。
- 不要写一个巨型万能 train/eval 脚本。
- 不要在 raw collector 中保存手工推断的 subtask、active_arm、phase。
- 不要在 raw collector 中手工计算并保存 TCP-cube distance、relative pose、progress 等派生字段。
- 不要用 post joint_pos 回填 18D action label。
- 不要把数据写到 `/home/qsh/dit`。
- 不要依赖旧服务器路径。
- 不要把旧仓库 clone 后当作运行时依赖，除非明确是 gitignored reference。
- 不要绕开官方 LeRobot 训练方式重写复杂训练 loop，除非官方入口不可用，并且必须在文档解释原因。
- 不要把 n_action_steps=8 当作正式默认值；它只能是诊断配置。
- 不要让 eval 静默覆盖 checkpoint 里的 horizon/n_action_steps。
- 不要让 collector、converter、train、eval 各自使用不同的 camera order、language instruction、joint order。

最终交付：
- `/home/qsh/dit` 新工作区
- 清晰 README
- conda/LeRobot setup 脚本
- setup check 脚本
- 通用 raw collector
- raw schema 文档
- raw validator
- state26 -> absjoint18 converter
- state26 -> absjoint18 官方 LeRobot MultiTask DiT train wrapper
- state26 -> absjoint18 open-loop eval
- state26 -> absjoint18 closed-loop eval
- known_pain_points.md
- smoke pipeline 可运行说明
- 所有大文件输出到 `/data/shared_folder/datasets/dit`