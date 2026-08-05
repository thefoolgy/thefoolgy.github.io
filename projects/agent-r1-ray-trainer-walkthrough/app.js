const SOURCES = {
  fit: {
    label: "fit 主循环",
    file: "./fit-source.py",
    start: 961,
    end: 1394,
    rangeLabel: "ray_trainer.py · RayAgentTrainer.fit() · L961-L1394",
    mainRanges: [[1000, 1005], [1016, 1025], [1036, 1061], [1069, 1148], [1154, 1189], [1191, 1241], [1269, 1275], [1331, 1373]],
    dynamicRanges: [[1016, 1022], [1036, 1038], [1040, 1051], [1059, 1061], [1069, 1140], [1348, 1357], [1366, 1370]],
  },
  helper: {
    label: "Dynamic helper",
    file: "./dynamic-helper-source.py",
    start: 85,
    end: 162,
    rangeLabel: "ray_trainer.py · Dynamic Sampling helpers · L85-L162",
    mainRanges: [[85, 162]],
    dynamicRanges: [[85, 162]],
  },
};

const INITIAL_STATE = {
  promptGroups: "0",
  rolloutRequests: "0",
  trajectories: "0",
  stepRows: "0",
  scoredRows: "0",
  promptUids: "—",
  trajectoryUids: "—",
  tensorKeys: "prompts",
  accumulated: "0 / 4 groups",
  optimizerBatch: "未就绪",
  advantages: "—",
  actor: "θ20",
};

const EVENT_DEFS = [
  {
    id: "config", phase: "准备", line: 1016, pass: "step 开始", impact: "control",
    title: "读取 Dynamic Sampling 开关",
    explanation: "每个 optimizer step 开始时读取 filter_groups 配置，并创建跨 refill 轮次保留的累积器。这里还没有训练数据。",
    patch: { accumulated: "0 / 4 groups", optimizerBatch: "等待生成" },
    delta: [["filter_groups_config", "不存在", "Hydra 配置对象"], ["accumulated_batch", "不存在", "None"]],
    why: "累积器必须放在 dataloader 循环外，否则 L1133 continue 之后，第一轮保留下来的 group 会丢失。",
  },
  {
    id: "raw-1", phase: "第 1 轮生成", line: 1036, pass: "第 1 轮", impact: "data",
    title: "把 4 条 dataset row 包装成 DataProto",
    explanation: "batch_dict 只有问题、标准答案和 reward 配置。DataProto.from_single_dict 把张量字段与 Python 对象字段放进统一跨 worker 协议。",
    patch: { promptGroups: "4 current", promptUids: "尚未分配", optimizerBatch: "等待生成", tensorKeys: "prompts" },
    delta: [["new_batch", "不存在", "DataProto · 4 rows"], ["当前 prompt", "0", "A, B, C, D"]],
    why: "从这里开始，页面中的“行数”指 DataProto 的物理 row 数，不等同于问题数或完整 trajectory 数。",
  },
  {
    id: "uid-1", phase: "第 1 轮生成", line: 1041, pass: "第 1 轮", impact: "data",
    title: "给每个 prompt group 分配 uid",
    explanation: "A、B、C、D 各自获得一个 uid。随后每题复制出的 n=4 条 rollout 仍共享这个 uid，GRPO 才知道哪些轨迹属于同一比较组。",
    patch: { promptUids: "A · B · C · D" },
    delta: [["non_tensor_batch.uid", "不存在", "[A, B, C, D]"], ["DataProto rows", "4", "4（行数不变）"]],
    why: "uid 是 prompt group 主键，不是 trajectory 主键。混淆两者会直接算错组内均值和标准差。",
  },
  {
    id: "repeat-1", phase: "第 1 轮生成", line: 1049, pass: "第 1 轮", impact: "data",
    title: "每题复制 n=4 个 rollout 请求",
    explanation: "这里只是生成请求从 4 行变成 16 行，还没有执行 Agent，也没有 trajectory_uids 或 step rows。",
    patch: { rolloutRequests: "16", trajectories: "0", stepRows: "0" },
    delta: [["gen_batch", "4 prompt rows", "16 request rows"], ["共享 uid", "每组 1 行", "每组 4 行"]],
    why: "GRPO 需要同一 prompt 的多个候选才能做组内相对比较。",
  },
  {
    id: "rollout-1", phase: "第 1 轮生成", line: 1057, pass: "第 1 轮", impact: "data",
    title: "AgentFlow 返回 16 条轨迹，展开成 22 个 step rows",
    explanation: "16 个请求各自运行多轮 search / observation / answer。因为轨迹步数不同，返回 DataProto 的物理行数是 22，而不是 16。",
    patch: { trajectories: "16", stepRows: "22 rollout rows", trajectoryUids: "16 ids across 22 rows", tensorKeys: "prompts · responses · masks" },
    delta: [["trajectory_uids", "不存在", "16 个 id，重复铺在 22 行"], ["num_steps", "不存在", "长度 16 的数组"], ["step rows", "0", "22"]],
    why: "这一步是 Agent loop 与 training loop 的边界。后面所有聚合都必须承认一条 trajectory 可能占多行。",
  },
  {
    id: "align-1", phase: "第 1 轮整理", line: 1070, pass: "第 1 轮", impact: "data",
    title: "先把原始 dataset 字段对齐到 16 条 trajectory",
    explanation: "question、ground_truth 等原始字段先按 n=4 复制。此时它们是一条 trajectory 一行，尚未对齐到多步 AgentFlow 输出。",
    patch: { tensorKeys: "dataset fields repeated to 16" },
    delta: [["new_batch dataset rows", "4", "16"], ["rollout step rows", "22", "22（另一份 DataProto）"]],
    why: "不能直接和 22 行 rollout union；两侧 row 数必须先完全对齐。",
  },
  {
    id: "num-steps-1", phase: "第 1 轮整理", line: 1073, pass: "第 1 轮", impact: "data",
    title: "取出每条 trajectory 的 step 数",
    explanation: "num_steps 的长度是 16，元素之和是 22。它告诉 sample_level_repeat 每条原始 trajectory row 应复制几次。",
    patch: { trajectoryUids: "16 ids · num_steps sum=22" },
    delta: [["gen_batch_output.meta_info", "含 num_steps", "弹出 num_steps"], ["num_steps", "未持有", "[1,1,…,2] · sum=22"]],
    why: "这是 trajectory 粒度和 step-row 粒度之间的转换表。",
  },
  {
    id: "flatten-1", phase: "第 1 轮整理", line: 1074, pass: "第 1 轮", impact: "data",
    title: "把 dataset trajectory rows 展开为 22 个 step rows",
    explanation: "sample_level_repeat(num_steps) 不是普通 repeat：每条 trajectory 按自己的步数复制，最终和 AgentFlow 输出逐行对应。",
    patch: { stepRows: "22 aligned rows", tensorKeys: "dataset fields · 22 aligned rows" },
    delta: [["new_batch rows", "16 trajectory rows", "22 step rows"], ["question / ground_truth", "每轨迹 1 份", "每 step 1 份"]],
    why: "如果在这里用固定次数 repeat，多步轨迹会错位，reward 可能对到另一条 response 上。",
  },
  {
    id: "union-1", phase: "第 1 轮整理", line: 1075, pass: "第 1 轮", impact: "data",
    title: "合并 dataset 字段与 rollout 字段",
    explanation: "union 后每个 step row 同时拥有 question/ground_truth、prompts/responses/masks、uid 和 trajectory_uids。",
    patch: { tensorKeys: "prompts · responses · masks · dataset fields" },
    delta: [["new_batch", "只有 dataset 字段", "dataset ∪ rollout"], ["粒度", "22 rows", "22 rows（字段变多）"]],
    why: "从这一行起，reward function 和后续 trainer 才拿到完整的一行训练样本。",
  },
  {
    id: "reward-1", phase: "第 1 轮打分", line: 1097, pass: "第 1 轮", impact: "data",
    title: "写入 22 行 token-level reward score",
    explanation: "reward function 对 step rows 打分，结果写进 token_level_scores。完整 trajectory 的回报还没有显式形成，需要 helper 按 trajectory_uids 求和。",
    patch: { scoredRows: "22", tensorKeys: "… + token_level_scores", optimizerBatch: "等待 DS gate" },
    delta: [["token_level_scores", "不存在", "[22, response_len]"], ["trajectory return", "未聚合", "仍分散在 step rows"]],
    why: "Dynamic Sampling 必须在 reward 之后，否则根本不知道同组 rollout 是否有差异。",
  },
  {
    id: "filter-1", phase: "第 1 轮筛选", line: 1105, pass: "第 1 轮", impact: "data",
    title: "第一次 Dynamic gate：4 组只保留 B、D",
    explanation: "helper 先把 22 个 row score 聚合成 16 个 trajectory return，再按 4 个 prompt uid 分组。A 全 1、C 全 0，标准差为 0；B、D 有差异，被完整保留。",
    patch: { promptGroups: "4 generated", trajectories: "16 generated", stepRows: "22 candidate rows", accumulated: "0 / 4 groups", optimizerBatch: "尚未就绪" },
    delta: [["filtered_batch", "不存在", "B,D · 14 step rows"], ["kept_prompt_uids", "不存在", "[B, D]"], ["acceptance", "未知", "2 / 4 = 50%"]],
    why: "筛选单位必须是完整 prompt group。只删掉某条失败 trajectory 会破坏 n=4 的组结构，也会让 GRPO 基线失真。",
    call: { label: "进入 filter_informative_prompt_groups()", source: "helper", line: 85 },
  },
  {
    id: "accumulate-1", phase: "第 1 轮筛选", line: 1112, pass: "第 1 轮", impact: "data",
    title: "把 B、D 的全部 step rows 放进累积池",
    explanation: "accumulated_batch 现在含 B、D 两组、8 条 trajectory、14 个 step rows；不是只有两个标量 reward。",
    patch: { promptGroups: "2 kept", trajectories: "8 kept", stepRows: "14 accumulated rows", scoredRows: "14", promptUids: "B · D", trajectoryUids: "8 ids across 14 rows", accumulated: "2 / 4 groups" },
    delta: [["accumulated_batch", "None", "DataProto · 14 rows"], ["accumulated_prompt_uids", "[]", "[B, D]"]],
    why: "保留整个 DataProto group，才能让后面的 log-prob、mask、advantage 仍然逐 row 对齐。",
  },
  {
    id: "refill-check", phase: "第 1 轮筛选", line: 1120, pass: "第 1 轮", impact: "control",
    title: "检查累积池是否达到 train_batch_size=4",
    explanation: "当前只有 2 组，因此条件为真。这里比较的是 prompt group 数，不是 trajectory 数或 step row 数。",
    patch: { optimizerBatch: "2 / 4 · 不足" },
    delta: [["len(accumulated_prompt_uids)", "2", "2"], ["分支结果", "未判断", "2 < 4 → True"]],
    why: "train_batch_size 在这段逻辑里是 prompt group budget。n=4 的 rollout 数是另一层 cardinality。",
  },
  {
    id: "continue-1", phase: "第 1 轮筛选", line: 1133, pass: "第 1 轮", impact: "control",
    title: "continue：不更新参数，回到 dataloader 补样",
    explanation: "这一轮生成结束，但 optimizer step 尚未完成。global_steps 不增加，B、D 留在 accumulated_batch，下一条 dataloader batch 会继续补。",
    patch: { optimizerBatch: "refill 中 · global step 仍为 21" },
    delta: [["控制流", "准备进入 old_log_prob", "跳回 L1025"], ["global_steps", "21", "21（不变）"], ["B,D", "已累积", "继续保留"]],
    why: "这就是 Dynamic Sampling 改变训练节奏的核心：generation batch 与 optimizer step 不再一一对应。",
  },
  {
    id: "raw-2", phase: "第 2 轮补样", line: 1036, pass: "第 2 轮 refill", impact: "data",
    title: "同一行第二次执行：读取 E、F、G、H",
    explanation: "这是 L1133 continue 后的下一次 dataloader 迭代。B、D 仍在累积池，新 batch 只负责产生新的候选 group。",
    patch: { promptGroups: "4 current · 8 generated total", rolloutRequests: "0 current", trajectories: "0 current", stepRows: "0 current", scoredRows: "0 current", promptUids: "E · F · G · H", trajectoryUids: "—", accumulated: "2 / 4 groups", optimizerBatch: "refill 第 2 轮" },
    delta: [["new_batch", "上一轮 A-D", "新一轮 E-H"], ["accumulated_batch", "B,D · 14 rows", "保持不变"]],
    why: "同一段源码在一个 optimizer step 内可以执行多次，所以只按静态代码块讲解会丢掉 refill 语义。",
  },
  {
    id: "rollout-2", phase: "第 2 轮补样", line: 1057, pass: "第 2 轮 refill", impact: "data",
    title: "第二轮再生成 16 条 trajectory",
    explanation: "E-H 的 16 条 rollout 展开为 19 个 step rows。累计生成成本已经是 8 个 prompt、32 条 trajectory，但仍只有一次待完成的 optimizer step。",
    patch: { rolloutRequests: "16 current · 32 total", trajectories: "16 current · 32 total", stepRows: "19 current rows", trajectoryUids: "16 new ids across 19 rows" },
    delta: [["本轮 step rows", "0", "19"], ["总生成 trajectories", "16", "32"]],
    why: "acceptance rate 越低，每次参数更新前花在 rollout 上的时间越多；因此必须监控 gen_batches。",
  },
  {
    id: "filter-2", phase: "第 2 轮补样", line: 1105, pass: "第 2 轮 refill", impact: "data",
    title: "第二次 Dynamic gate：保留 E、F",
    explanation: "E、F 的 n=4 回报有差异；G 全 0、H 全 1 被拒绝。本轮 filtered_batch 含 9 个 step rows。",
    patch: { scoredRows: "19 current", promptGroups: "4 current · 8 generated total", trajectories: "16 current · 32 total", stepRows: "19 candidate rows" },
    delta: [["filtered_batch", "不存在", "E,F · 9 step rows"], ["累计 acceptance", "2 / 4", "4 / 8 = 50%"]],
    why: "helper 每一轮只判断当前 new_batch；跨轮合并由 fit() 的 accumulated_batch 负责。",
    call: { label: "再次进入 Dynamic helper", source: "helper", line: 85 },
  },
  {
    id: "concat-2", phase: "凑齐 batch", line: 1115, pass: "第 2 轮 refill", impact: "data",
    title: "DataProto.concat：B、D 与 E、F 合并",
    explanation: "旧累积池 14 行和新 filtered_batch 9 行拼成 23 行。所有 tensor、non-tensor 字段和 meta_info 必须兼容。",
    patch: { promptGroups: "4 kept", trajectories: "16 kept", stepRows: "23 accumulated rows", scoredRows: "23", promptUids: "B · D · E · F", trajectoryUids: "16 ids across 23 rows", accumulated: "4 / 4 groups", optimizerBatch: "已凑齐，待选择" },
    delta: [["accumulated_batch", "B,D · 14 rows", "B,D,E,F · 23 rows"], ["accumulated_prompt_uids", "[B,D]", "[B,D,E,F]"]],
    why: "之前遇到 reward_extra_keys 冲突，正是在 concat 多个 DataProto 时 meta_info 不一致；这行是跨轮数据协议的压力点。",
  },
  {
    id: "optimizer-batch", phase: "凑齐 batch", line: 1136, pass: "第 2 轮 refill", impact: "data",
    title: "选出完整的 4 个 prompt group 作为 optimizer batch",
    explanation: "select_prompt_groups 按 uid 选择 B、D、E、F 的全部 23 个 step rows。到这里，batch 才成为后续 PPO/GRPO 共同消费的固定训练批次。",
    patch: { optimizerBatch: "READY · B,D,E,F", promptGroups: "4 selected", trajectories: "16 selected", stepRows: "23 selected rows" },
    delta: [["batch", "上一 optimizer step 的变量", "DataProto · 23 rows"], ["selected_prompt_uids", "未定义", "[B,D,E,F]"]],
    why: "Dynamic Sampling 到此结束。它不直接计算 advantage，只决定哪些完整 group 有资格进入 advantage 计算。",
    call: { label: "打开 select_prompt_groups()", source: "helper", line: 158 },
  },
  {
    id: "old-log-prob", phase: "策略统计", line: 1166, pass: "optimizer batch", impact: "data",
    title: "只为最终 23 行计算 old log-prob",
    explanation: "被拒绝的 A、C、G、H 不再做 actor forward。π_old 是本次 PPO update 的固定概率锚点。",
    patch: { tensorKeys: "… + token_level_scores + old_log_probs" },
    delta: [["old_log_probs", "不存在", "[23, response_len]"], ["计算对象", "全部生成候选", "仅 B,D,E,F"]],
    why: "把 gate 放在 old log-prob 之前可以省掉无效 group 的训练侧 forward 成本。",
  },
  {
    id: "advantage", phase: "GRPO 信号", line: 1233, pass: "optimizer batch", impact: "data",
    title: "按 uid 计算组内相对 advantage",
    explanation: "B、D、E、F 每组都有不同 trajectory return，因此组内标准化后产生正负 advantage，并广播回同一 trajectory 的 step/action tokens。",
    patch: { advantages: "4 / 4 groups 有非零组内信号", tensorKeys: "… + token_level_rewards + advantages + returns" },
    delta: [["advantages", "不存在", "[23, response_len]"], ["zero-advantage groups", "候选中 4 / 8", "optimizer batch 中 0 / 4"]],
    why: "Dynamic Sampling 真正改善的是送到这一行的数据分布，而不是修改 GRPO 公式本身。",
  },
  {
    id: "actor-update", phase: "参数更新", line: 1273, pass: "optimizer batch", impact: "data",
    title: "Ray worker 用 23 个 step rows 更新 Actor",
    explanation: "driver 把带有 masks、old_log_probs、rewards 和 advantages 的 DataProto 发给 actor worker；真正的 forward、loss.backward 与 optimizer.step 在 worker 内执行。",
    patch: { actor: "θ21", optimizerBatch: "已消费", advantages: "已用于 policy loss" },
    delta: [["Actor 参数", "θ20", "θ21"], ["global_steps", "21", "仍为 21（稍后递增）"]],
    why: "一个 optimizer step 到这里才真的发生。前面两轮 rollout 都只是为它准备有学习信号的数据。",
  },
  {
    id: "metrics", phase: "收尾", line: 1348, pass: "step 收尾", impact: "data",
    title: "记录 Dynamic Sampling 成本与接受率",
    explanation: "本例 gen_batches=2、generated_prompts=8、kept_prompts=4、acceptance_rate=0.5。只看最终 reward 看不到为补样付出的 rollout 成本。",
    patch: { optimizerBatch: "已记录 metrics", accumulated: "4 / 4 · 即将 reset" },
    delta: [["dynamic_sampling/gen_batches", "不存在", "2"], ["acceptance_rate", "不存在", "0.5"], ["selected_prompts", "不存在", "4"]],
    why: "matched ablation 至少要同时比较 optimizer steps、总 rollout 数和 GPU 时间，不能只按 step 数宣称更高效。",
  },
  {
    id: "reset", phase: "收尾", line: 1366, pass: "step 收尾", impact: "data",
    title: "清空 refill 状态，准备下一个 optimizer step",
    explanation: "timing、accumulated_batch、prompt uid 列表和统计计数全部重置；Actor θ21 保留。随后 global_steps 才从 21 变成 22。",
    patch: { promptGroups: "0", rolloutRequests: "0", trajectories: "0", stepRows: "0", scoredRows: "0", promptUids: "—", trajectoryUids: "—", tensorKeys: "prompts", accumulated: "0 / 4 groups", optimizerBatch: "等待下一 step", advantages: "—", actor: "θ21" },
    delta: [["accumulated_batch", "23 rows", "None"], ["num_gen_batches", "2", "0"], ["dynamic_sampling_totals", "本 step 统计", "清空"]],
    why: "reset 必须发生在 logger.log 之后，否则记录到的 acceptance rate 和 gen_batches 会变成 0。",
  },
];

const LINE_DETAILS = {
  fit: {
    1001: ["创建训练进度条", "这是 UI/日志状态，不改变 DataProto。", "control"],
    1003: ["说明 global step 从 1 开始", "注释只解释计数约定，不产生运行时数据变化。", "none"],
    1004: ["推进起始 global step", "从 checkpoint 恢复值加 1；尚未执行 optimizer update。", "control"],
    1017: ["把配置转成布尔开关", "只有 filter_groups 存在且 enable=true，才进入 L1103 的筛选分支。", "control"],
    1018: ["创建跨 refill 的计时器", "defaultdict(float) 允许第二轮 generation 时间累加到同一个 optimizer step。", "data"],
    1019: ["初始化空累积 DataProto", "第一轮通过 gate 后，它会指向 filtered_batch。", "data"],
    1020: ["初始化 prompt uid 累积列表", "这里数的是保留的 prompt group，不是 step rows。", "data"],
    1021: ["初始化 generation batch 计数", "每次 dataloader 补采加 1；最终写入 dynamic_sampling/gen_batches。", "data"],
    1022: ["初始化 Dynamic Sampling 统计", "跨 refill 累计 generated、kept、rejected prompt 和 row 数。", "data"],
    1025: ["进入 dataloader 循环", "无 Dynamic Sampling 时通常一次循环对应一次 update；开启后，continue 可能让多次循环共同组成一次 update。", "control"],
    1038: ["generation batch 计数加一", "第一次为 1，refill 后第二次为 2。它不是 global_steps。", "data"],
    1045: ["裁出 rollout 所需字段", "避免把训练专用 tensor 发送给 AgentFlow。", "data"],
    1048: ["把 global step 写入 trace meta", "工具轨迹和 timing 日志可追溯到同一个 optimizer step。", "data"],
    1050: ["读取 rollout.n", "本例 n=4，因此 4 个 prompt 生成 16 个请求。", "none"],
    1053: ["判断是否最后一个 optimizer step", "影响验证和保存，不改变当前训练 batch。", "control"],
    1059: ["累加 AgentFlow 内部 timing", "refill 会再次进入这里，所以必须 +=，不能覆盖第一轮耗时。", "data"],
    1061: ["从返回值移除 timing meta", "避免后续 DataProto.concat 因每轮 timing 不同而触发 meta_info 冲突。", "data"],
    1077: ["检查 response_mask 是否存在", "AgentFlow 未提供时才在下一行补算。", "control"],
    1078: ["构造 response_mask", "只有 response/action token 为 1；prompt 和 padding token 不进入 policy loss。", "data"],
    1080: ["进入 reward 计时段", "reward 耗时与 rollout、old_log_prob、update_actor 分开统计。", "control"],
    1082: ["决定是否调用神经 Reward Model", "HotpotQA 规则 reward 常直接走 _compute_or_extract_reward。", "control"],
    1094: ["同步计算或提取训练 reward", "返回 token-level reward tensor 与额外指标字典。", "data"],
    1098: ["判断是否存在 reward extras", "例如 EM、F1、format_valid；没有则不更新 non_tensor_batch。", "control"],
    1099: ["把 reward extras 放进 DataProto", "每个 key 都必须与 row 数对齐，跨 refill concat 时 schema 也必须一致。", "data"],
    1103: ["进入 Dynamic Sampling 分支", "关闭开关时直接执行 L1138：batch = new_batch。", "control"],
    1104: ["选择用于筛选的 metric", "seq_reward 使用 KL 前的 token_level_scores；seq_final_reward 使用 token_level_rewards。", "data"],
    1108: ["遍历本轮 filter 统计", "generated_prompts、kept_prompts 等被累加到 optimizer-step 级统计。", "control"],
    1111: ["只有存在通过的 group 才累积", "如果本轮全被拒绝，accumulated_batch 保持原样。", "control"],
    1115: ["跨 refill 拼接 DataProto", "两侧 tensor/non-tensor schema 和 meta_info 必须兼容。", "data"],
    1117: ["追加保留的 prompt uid", "这个列表决定是否已经凑够 train_batch_size。", "data"],
    1119: ["读取目标 prompt batch size", "本例为 4；真正的 trajectory budget 是 4×n=16。", "data"],
    1121: ["读取最大补采轮数", "0 表示不限；正数可避免 acceptance 极低时无限循环。", "data"],
    1127: ["检查是否超过 refill 上限", "达到上限仍未凑满时明确报错，不用残缺 group 继续训练。", "control"],
    1135: ["截取前 train_batch_size 个 uid", "如果最后一轮保留过多 group，这里只选需要的前 4 组。", "data"],
    1137: ["Dynamic Sampling 关闭分支", "关闭时当前 new_batch 原样成为 optimizer batch。", "control"],
    1138: ["不筛选：直接使用当前 batch", "这是 matched baseline 的数据路径。", "data"],
    1140: ["Dynamic gate 的结束边界", "从这里开始，代码只消费已经凑齐的固定 optimizer batch。", "none"],
    1141: ["记录每行有效 token 数", "attention_mask 按序列求和，供 balance、throughput 和日志使用。", "data"],
    1144: ["pad 到 world size 可整除", "补的物理 row 由 sample_mask 排除，不贡献 reward 和 loss。", "data"],
    1147: ["判断是否做 DP token balance", "长短 step 差异大时，按 token 数重排行可以减少某张卡拖尾。", "control"],
    1148: ["重排 optimizer batch", "uid 和 trajectory_uids 会随 row 一起移动，语义分组不依赖行顺序。", "data"],
    1154: ["读取 rollout correction 配置", "它处理 rollout engine 与 training engine 的 policy mismatch，不是 Dynamic Sampling。", "data"],
    1155: ["选择 bypass 或 recompute old log-prob", "这是 PPO 概率锚点的实现选择。", "control"],
    1182: ["把 old_log_probs 合并回 batch", "之后 PPO ratio 才能比较当前策略与固定 π_old。", "data"],
    1189: ["断言 old_log_probs 已存在", "这是进入 advantage 和 Actor update 前的数据契约。", "control"],
    1191: ["判断是否使用 reference policy", "Reference KL 限制策略漂移；它与筛掉零方差 group 是两个正交机制。", "control"],
    1194: ["计算 reference log-prob", "只对最终 optimizer batch 计算。", "data"],
    1203: ["进入 advantage 计时段", "reward KL、rollout correction 和 advantage 都归入该阶段。", "control"],
    1205: ["决定是否把 KL 放进 reward", "开启时 token_level_rewards = score - β·KL；关闭时直接复制 scores。", "control"],
    1211: ["复制原始 score 为最终 token reward", "本行没有 Reference KL 惩罚。", "data"],
    1213: ["rollout correction 说明", "IS/rejection correction 针对采样策略偏差，不负责处理组内 reward 全相同。", "none"],
    1224: ["计算 rollout correction", "可能加入 IS 权重或 token rejection mask。", "data"],
    1228: ["advantage 计算边界", "driver 负责轻量的组内聚合，GPU worker 负责模型计算。", "none"],
    1229: ["读取 GRPO 是否除以组内标准差", "关闭后可只减均值；仍然需要同组 reward 存在差异才有信号。", "data"],
    1241: ["compute_advantage 返回完整 batch", "新增 advantages 与 returns，并保持原 row 对齐。", "data"],
    1269: ["Critic warmup 边界", "GRPO 通常不使用 critic，但 Actor 仍受 critic_warmup 配置门控。", "control"],
    1270: ["判断是否允许更新 Actor", "达到 warmup step 后才执行下一行的 worker RPC。", "control"],
    1272: ["进入 Actor update 计时段", "这里记录的是训练侧更新耗时，不是 rollout generation。", "control"],
    1339: ["移除 padding rows 后统计", "Data metrics 只看 sample_mask 有效的真实 step rows。", "data"],
    1341: ["汇总 reward/advantage 数据指标", "使用最终 B,D,E,F optimizer batch，而不是所有生成候选。", "data"],
    1349: ["记录本 step 生成了几批", "本例为 2；它揭示 refill 带来的额外 rollout 成本。", "data"],
    1351: ["写入累计筛选统计", "包括 generated、kept、rejected prompt 和 kept rows。", "data"],
    1354: ["计算 acceptance rate", "kept_prompts / generated_prompts；本例 4/8=0.5。", "data"],
    1357: ["记录最终选择的 prompt 数", "它应等于 train_batch_size，而非生成总数。", "data"],
    1364: ["把完整 step metrics 写入 logger", "必须先写日志再 reset Dynamic Sampling 累积器。", "data"],
    1367: ["释放累积 DataProto", "下一个 optimizer step 不可复用本 step 的 group。", "data"],
    1368: ["清空 prompt uid 列表", "下个 step 从 0/4 重新补。", "data"],
    1369: ["清零 generation batch 计数", "下一 optimizer step 的第一轮重新记为 1。", "data"],
    1370: ["清空筛选统计", "避免把前一个 step 的 generated/kept 数累计进下一 step。", "data"],
    1372: ["进度条完成一次 update", "虽然生成了两批数据，进度只增加 1。", "control"],
    1373: ["global_steps 加一", "Actor 已更新、指标已记录后，step 21 才推进为 22。", "control"],
  },
  helper: {
    85: ["定义完整 group 筛选器", "输入是已经完成 rollout、flatten 和 reward 的 AgentFlow DataProto。输出仍是 DataProto，但只包含有组内差异的完整 prompt groups。", "data"],
    90: ["函数契约：保留完整 trajectory return 不全相同的组", "关键字是 complete trajectory returns，不是单个 step reward。", "none"],
    92: ["说明 AgentFlow 的多步展开", "一条 rollout 会占多个 step rows，所以不能直接在 row 上计算组内标准差。", "none"],
    96: ["选择原始 seq_reward", "默认对 token_level_scores 每行沿 token 维求和，先得到每个 step row 的标量。", "control"],
    97: ["Token score → row metric", "输入 [step_rows, response_len]，输出长度为 step_rows 的 NumPy 数组。", "data"],
    98: ["可选最终 reward 分支", "seq_final_reward 读取 token_level_rewards，可能已经包含 KL penalty。", "control"],
    100: ["支持 non-tensor 自定义 metric", "如果 reward extra 已按 row 写入 non_tensor_batch，也可以拿它筛选。", "control"],
    108: ["读取 prompt uid", "它把 n 条 trajectory 归为同一个 GRPO group。", "data"],
    109: ["读取 trajectory uid", "它把同一 rollout 展开的多个 step rows 重新聚合回来。", "data"],
    110: ["检查两层主键是否齐全", "少任意一个都无法正确完成 step→trajectory→prompt 的两级聚合。", "control"],
    112: ["检查所有 row 字段严格对齐", "DataProto 行数、uid、trajectory_uids、row metric 长度必须完全相同。", "control"],
    115: ["创建 trajectory return 累加表", "key 是 trajectory_uid，value 最终会成为该轨迹所有 step row reward 的和。", "data"],
    121: ["逐个 step row 遍历三列对齐数据", "每次循环同时拿 prompt_uid、trajectory_uid 和该 row 的 reward。", "control"],
    125: ["首次见到一条 trajectory", "初始化 return=0，并记录它属于哪个 prompt group。", "data"],
    129: ["防止同一 trajectory 跨 prompt", "出现这种情况说明上游 uid 对齐损坏，必须报错而不是静默聚合。", "control"],
    131: ["Step row reward 累加成 trajectory return", "这是最关键的一行：同一 trajectory 的 search/answer 多行先相加，才得到一个完整 rollout 回报。", "data"],
    133: ["创建 prompt → returns 的分组表", "value 将是同一个 prompt 的 n 个完整 trajectory returns。", "data"],
    134: ["按 trajectory 首次出现顺序遍历", "保证 group 和返回结果顺序稳定。", "control"],
    135: ["Trajectory return 放回 prompt group", "数据结构从 trajectory_totals[T] 变成 prompt_returns[P]=[R1,R2,R3,R4]。", "data"],
    137: ["开始构造保留 uid 列表", "筛选结果仍使用 prompt uid 表示，而不是 row index。", "data"],
    140: ["标准差 gate", "n>1 时仅保留 std>min_std 的 group。全 1 和全 0 都被拒绝；有高有低才保留。", "data"],
    142: ["把保留列表转成 set", "仅用于下一行快速判断每个 step row 的 prompt 是否通过。", "data"],
    143: ["找回通过 group 的全部 step row 下标", "同一 prompt 的所有 trajectory、所有步骤一起保留，没有只挑好 trajectory。", "data"],
    144: ["按 row index 切出 filtered DataProto", "TensorDict 和 non_tensor_batch 会同步切片，row 对齐关系保持不变。", "data"],
    146: ["统计本轮生成 prompt 数", "这是当前 generation batch 的 4 组，不含之前累积池。", "data"],
    148: ["构造可观测筛选统计", "用于最终 acceptance_rate、生成成本和诊断日志。", "data"],
    155: ["返回三件东西", "filtered DataProto 供 concat，kept_prompt_uids 供计数/选择，stats 供日志。", "data"],
    158: ["定义完整 prompt group 选择器", "当累积池达到目标后，按最终 uid 列表选择完整 step rows。", "data"],
    160: ["选择 uid 转成 set", "保证逐 row 判断是常数时间。", "data"],
    161: ["收集最终 group 的全部 row 下标", "选择单位仍是 prompt uid，输出物理上仍是 step rows。", "data"],
    162: ["返回固定 optimizer batch", "fit() 后续只消费这份 DataProto。", "data"],
  },
};

const GROUPS = {
  first: [
    ["A", "1, 1, 1, 1", "0.000", "reject"],
    ["B", "1, .6, 0, .2", "0.384", "keep"],
    ["C", "0, 0, 0, 0", "0.000", "reject"],
    ["D", ".8, .8, .2, .4", "0.260", "keep"],
  ],
  second: [
    ["E", "1, .4, .2, 0", "0.374", "keep"],
    ["F", ".9, .7, .1, 0", "0.383", "keep"],
    ["G", "0, 0, 0, 0", "0.000", "reject"],
    ["H", "1, 1, 1, 1", "0.000", "reject"],
  ],
  selected: [
    ["B", "1, .6, 0, .2", "0.384", "selected"],
    ["D", ".8, .8, .2, .4", "0.260", "selected"],
    ["E", "1, .4, .2, 0", "0.374", "selected"],
    ["F", ".9, .7, .1, 0", "0.383", "selected"],
  ],
};

let sourceData = {};
let activeSource = "fit";
let activeView = "mainline";
let selectedLine = 1036;
let selectedEventId = "raw-1";

function cloneState(state) {
  return { ...state };
}

function buildEvents() {
  let state = cloneState(INITIAL_STATE);
  return EVENT_DEFS.map((event, index) => {
    const before = cloneState(state);
    state = { ...state, ...event.patch };
    return { ...event, index, before, after: cloneState(state) };
  });
}

const runtimeEvents = buildEvents();
const eventsById = new Map(runtimeEvents.map((event) => [event.id, event]));
const BEFORE_FIRST_EVENT = {
  id: "before-first-event",
  index: -1,
  line: 961,
  pass: "进入 fit()",
  impact: "none",
  before: cloneState(INITIAL_STATE),
  after: cloneState(INITIAL_STATE),
  delta: null,
  why: "此时还没有从 dataloader 取数据，也没有执行 Dynamic Sampling。",
};

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[char]);
}

function inRanges(line, ranges) {
  return ranges.some(([start, end]) => line >= start && line <= end);
}

function isDynamicLine(source, line) {
  return inRanges(line, SOURCES[source].dynamicRanges);
}

function impactForLine(source, line) {
  const detail = LINE_DETAILS[source]?.[line];
  if (detail) return detail[2];
  const text = getLineText(source, line).trim();
  if (!text || text.startsWith("#") || text.startsWith('"""')) return "none";
  if (/^(if|elif|else:|for|while|with|try:|except|return|continue|raise|assert)\b/.test(text)) return "control";
  if (text.includes("=") || text.includes(".update(") || text.includes(".append(") || text.includes(".extend(")) return "data";
  return "none";
}

function getLineText(source, line) {
  const data = sourceData[source];
  if (!data) return "";
  return data[line - SOURCES[source].start] ?? "";
}

function classifyCode(text) {
  const trimmed = text.trim();
  if (!trimmed) return "blank";
  if (trimmed.startsWith("#") || trimmed.startsWith('"""')) return "comment";
  if (/^(def|for|if|elif|else|while|with|return|continue|raise|assert|try|except|from|import)\b/.test(trimmed)) return "keyword";
  if (/^[A-Za-z_][\w.\[\]"']*\s*=/.test(trimmed)) return "assignment";
  return "plain";
}

function renderTrace() {
  const target = document.getElementById("trace-list");
  let phase = "";
  target.innerHTML = runtimeEvents.map((event) => {
    const phaseHeading = event.phase !== phase ? `<div class="trace-phase">${escapeHtml(event.phase)}</div>` : "";
    phase = event.phase;
    return `${phaseHeading}<button class="trace-event${event.id === selectedEventId ? " active" : ""}" data-event="${event.id}">
      <span class="trace-line">L${event.line}</span>
      <span class="trace-copy"><b>${escapeHtml(event.title)}</b><small>${escapeHtml(event.pass)}</small></span>
      <i class="impact-dot ${event.impact}"></i>
    </button>`;
  }).join("");
  target.querySelectorAll("[data-event]").forEach((button) => {
    button.addEventListener("click", () => selectEvent(button.dataset.event, true));
  });
}

function sourceLineNumbers(source) {
  const config = SOURCES[source];
  return Array.from({ length: config.end - config.start + 1 }, (_, index) => config.start + index);
}

function visibleLineNumbers(source) {
  const lines = sourceLineNumbers(source);
  if (activeView === "full") return lines;
  return lines.filter((line) => inRanges(line, SOURCES[source].mainRanges));
}

function renderSource() {
  const target = document.getElementById("source-code");
  const lines = visibleLineNumbers(activeSource);
  const fragments = [];
  let previous = null;
  for (const line of lines) {
    if (previous !== null && line > previous + 1) {
      fragments.push(`<button class="code-gap" data-expand-line="${previous + 1}" title="切换到完整源码">··· 隐藏 L${previous + 1}–L${line - 1} 的支线代码 ···</button>`);
    }
    const text = getLineText(activeSource, line);
    const dynamic = isDynamicLine(activeSource, line);
    const impact = impactForLine(activeSource, line);
    const eventCount = activeSource === "fit" ? runtimeEvents.filter((event) => event.line === line).length : 0;
    fragments.push(`<button class="code-line${line === selectedLine ? " active" : ""}${dynamic ? " dynamic-line" : ""}" data-line="${line}" role="option" aria-selected="${line === selectedLine}">
      <span class="line-number">${line}</span>
      <span class="line-marker ${dynamic ? "dynamic" : impact}">${dynamic ? "DS" : impact === "data" ? "DATA" : impact === "control" ? "CTRL" : ""}</span>
      <code class="code-text ${classifyCode(text)}">${escapeHtml(text || " ")}</code>
      ${eventCount > 1 ? `<span class="repeat-count">×${eventCount}</span>` : ""}
    </button>`);
    previous = line;
  }
  target.innerHTML = fragments.join("");
  target.querySelectorAll(".code-line").forEach((row) => row.addEventListener("click", () => selectLine(activeSource, Number(row.dataset.line), true)));
  target.querySelectorAll(".code-gap").forEach((button) => button.addEventListener("click", () => {
    activeView = "full";
    updateViewControls();
    renderSource();
    selectLine(activeSource, Number(button.dataset.expandLine), true);
  }));
}

function autoDetail(source, line) {
  const text = getLineText(source, line).trim();
  if (!text) return ["空行：视觉分隔，不改变运行状态", "这一行没有独立 Python 语义；右侧账本保持在上一条有效语句之后。", "none"];
  if (text.startsWith("#") || text.startsWith('"""')) return ["注释：不改变运行状态", text.replace(/^#+\s*/, ""), "none"];
  if (text === "continue") return ["跳回当前循环的下一次迭代", "当前语句只改变控制流，不直接改写 DataProto。", "control"];
  if (text.startsWith("if ") || text.startsWith("elif ") || text === "else:") return ["判断当前分支", `条件语句：${text}`, "control"];
  if (text.startsWith("for ")) return ["遍历当前集合", `循环语句：${text}`, "control"];
  if (text.startsWith("with ")) return ["进入上下文管理器", `通常用于计时或资源管理：${text}`, "control"];
  if (text.startsWith("return")) return ["返回当前结果", text, "control"];
  const assignment = text.match(/^([^=]+?)\s*=\s*(.+)$/);
  if (assignment) return [`更新 ${assignment[1].trim()}`, `执行赋值：${text}`, "data"];
  return ["执行当前 Python 语句", text, impactForLine(source, line)];
}

function detailForLine(source, line) {
  return LINE_DETAILS[source]?.[line] || autoDetail(source, line);
}

function primaryEventForLine(line) {
  const exact = runtimeEvents.filter((event) => event.line === line);
  if (exact.length) return exact[0];
  if (line < 1036) {
    return [...runtimeEvents].filter((event) => event.line <= line && event.line < 1036).at(-1) || BEFORE_FIRST_EVENT;
  }
  if (line <= 1133) {
    return [...runtimeEvents].filter((event) => event.index <= eventsById.get("continue-1").index && event.line <= line).at(-1) || eventsById.get("raw-1");
  }
  const postRefill = runtimeEvents.filter((event) => event.index >= eventsById.get("concat-2").index && event.line <= line);
  return postRefill.at(-1) || eventsById.get("concat-2");
}

function helperContext(line) {
  const base = eventsById.get("filter-1");
  const before = cloneState(base.before);
  let after = cloneState(before);
  if (line >= 97) after.scoredRows = "22 row metrics";
  if (line >= 108) after.promptUids = "22 row labels → 4 unique";
  if (line >= 109) after.trajectoryUids = "22 row labels → 16 unique";
  if (line >= 131) after.trajectories = "16 trajectory totals";
  if (line >= 135) after.promptGroups = "4 groups of 4 returns";
  if (line >= 140) after.accumulated = "candidate keep: B,D";
  if (line >= 144) {
    after.promptGroups = "2 filtered";
    after.trajectories = "8 filtered";
    after.stepRows = "14 filtered rows";
    after.promptUids = "B · D";
  }
  return {
    id: `helper-${line}`, index: base.index, line, pass: "helper · 第 1 轮示例", impact: impactForLine("helper", line),
    before, after, delta: null,
    why: "helper 的主线始终是两级聚合：step row 先按 trajectory_uids 求和，再按 uid 比较同题的多个 trajectory。",
  };
}

function contextForSelection(source, line, preferredEventId = null) {
  if (source === "helper") return helperContext(line);
  if (preferredEventId) {
    const preferred = eventsById.get(preferredEventId);
    if (preferred && preferred.line === line) return preferred;
  }
  return primaryEventForLine(line);
}

function stateRows(state) {
  return [
    ["prompt groups", state.promptGroups],
    ["rollout requests", state.rolloutRequests],
    ["trajectories", state.trajectories],
    ["AgentFlow step rows", state.stepRows],
    ["scored rows", state.scoredRows],
    ["uid", state.promptUids],
    ["trajectory_uids", state.trajectoryUids],
    ["tensor keys", state.tensorKeys],
    ["DS accumulator", state.accumulated],
    ["optimizer batch", state.optimizerBatch],
    ["advantages", state.advantages],
    ["Actor", state.actor],
  ];
}

function renderCardinality(state) {
  const parseCount = (value) => Number(String(value).match(/\d+/)?.[0] || 0);
  const items = [
    ["groups", parseCount(state.promptGroups), 8],
    ["trajectories", parseCount(state.trajectories), 32],
    ["step rows", parseCount(state.stepRows), 23],
    ["scored", parseCount(state.scoredRows), 23],
  ];
  document.getElementById("cardinality").innerHTML = items.map(([label, count, max]) => `
    <div class="count-item"><span>${label}</span><b>${count}</b><i><em style="width:${Math.min(100, (count / max) * 100)}%"></em></i></div>`).join("");
}

function renderLedger(before, after) {
  const rows = stateRows(after);
  document.getElementById("state-ledger").innerHTML = rows.map(([key, value], index) => {
    const previous = stateRows(before)[index][1];
    const changed = previous !== value;
    return `<div class="ledger-row${changed ? " changed" : ""}"><code>${escapeHtml(key)}</code><span>${escapeHtml(value)}</span></div>`;
  }).join("");
  renderCardinality(after);
}

function renderDelta(context, detail, line) {
  let delta = context.line === line && context.delta ? context.delta : null;
  if (!delta) {
    const impact = detail[2];
    delta = impact === "data"
      ? [["当前语句", "尚未执行", getLineText(activeSource, line).trim() || "空行"], ["账本", "上一状态", "查看下方高亮字段"]]
      : [["DataProto", "保持不变", "保持不变"], ["控制流", "到达本行", impact === "control" ? "执行判断/跳转" : "继续下一行"]];
  }
  document.getElementById("local-delta").innerHTML = delta.map(([key, before, after]) => `
    <div class="delta-row"><code>${escapeHtml(key)}</code><span>${escapeHtml(before)}</span><i>→</i><strong>${escapeHtml(after)}</strong></div>`).join("");
}

function groupSetForContext(context) {
  if (activeSource === "helper") return GROUPS.first;
  if (context.index >= eventsById.get("optimizer-batch").index) return GROUPS.selected;
  if (context.index >= eventsById.get("raw-2").index) return GROUPS.second;
  return GROUPS.first;
}

function renderGroups(context) {
  const groups = groupSetForContext(context);
  document.getElementById("group-table").innerHTML = `
    <div class="group-head"><span>uid</span><span>4 trajectory returns</span><span>std</span><span>decision</span></div>
    ${groups.map(([uid, rewards, std, decision]) => `<div class="group-row ${decision}"><b>${uid}</b><code>[${rewards}]</code><span>${std}</span><strong>${decision}</strong></div>`).join("")}`;
}

function renderOccurrences(source, line, context) {
  const target = document.getElementById("occurrence-switch");
  if (source !== "fit") {
    target.innerHTML = "";
    return;
  }
  const occurrences = runtimeEvents.filter((event) => event.line === line);
  if (occurrences.length <= 1) {
    target.innerHTML = "";
    return;
  }
  target.innerHTML = `<span>本行在同一 optimizer step 执行 ${occurrences.length} 次：</span>${occurrences.map((event, index) => `<button class="${event.id === context.id ? "active" : ""}" data-occurrence="${event.id}">第 ${index + 1} 次</button>`).join("")}`;
  target.querySelectorAll("[data-occurrence]").forEach((button) => button.addEventListener("click", () => selectEvent(button.dataset.occurrence, false)));
}

function renderInspector(source, line, preferredEventId = null) {
  const context = contextForSelection(source, line, preferredEventId);
  const lineDetail = detailForLine(source, line);
  const detail = source === "fit" && context.line === line && context.title
    ? [context.title, context.explanation, context.impact]
    : lineDetail;
  const impact = detail[2] || context.impact || "none";
  document.getElementById("line-badge").textContent = `L${line}`;
  const impactBadge = document.getElementById("impact-badge");
  impactBadge.className = `impact-badge ${impact}`;
  impactBadge.textContent = impact === "data" ? "DATA CHANGE" : impact === "control" ? "CONTROL FLOW" : "NO DATA CHANGE";
  document.getElementById("pass-badge").textContent = context.pass || SOURCES[source].label;
  document.getElementById("line-title").textContent = detail[0];
  document.getElementById("line-explanation").textContent = detail[1];
  renderOccurrences(source, line, context);

  const call = context.line === line ? context.call : null;
  const callTarget = document.getElementById("call-chain");
  callTarget.innerHTML = call ? `<button data-call-source="${call.source}" data-call-line="${call.line}">${escapeHtml(call.label)} <span>→</span></button>` : "";
  callTarget.querySelector("button")?.addEventListener("click", (event) => selectLine(event.currentTarget.dataset.callSource, Number(event.currentTarget.dataset.callLine), true));

  renderDelta(context, detail, line);
  renderLedger(context.before, context.after);
  renderGroups(context);
  document.getElementById("why-text").textContent = context.line === line && context.why ? context.why : (impact === "none" ? "这一行不独立改变数据。主线状态沿用上一条有效语句；选择相邻的 DATA 行可以看到真实 before/after。" : "这一行位于当前数据转换内。下方账本高亮的是执行到这里已经发生变化的字段。 ");
  document.getElementById("ledger-caption").textContent = `${context.pass || "当前路径"} · 执行到 L${line}`;
  document.getElementById("trace-progress").textContent = `${context.index + 1} / ${runtimeEvents.length}`;
}

function updateTraceSelection(eventId) {
  document.querySelectorAll(".trace-event").forEach((button) => button.classList.toggle("active", button.dataset.event === eventId));
  const active = document.querySelector(`.trace-event[data-event="${eventId}"]`);
  if (active) {
    const container = document.getElementById("trace-list");
    const top = active.offsetTop;
    if (top < container.scrollTop || top + active.offsetHeight > container.scrollTop + container.clientHeight) {
      container.scrollTo({ top: Math.max(0, top - container.clientHeight / 3), behavior: "smooth" });
    }
  }
}

function updateSourceSelection(scroll = false) {
  document.querySelectorAll(".code-line").forEach((row) => {
    const active = Number(row.dataset.line) === selectedLine;
    row.classList.toggle("active", active);
    row.setAttribute("aria-selected", String(active));
  });
  const active = document.querySelector(`.code-line[data-line="${selectedLine}"]`);
  if (scroll && active) {
    const container = document.getElementById("code-scroll");
    const top = active.offsetTop;
    container.scrollTo({ top: Math.max(0, top - container.clientHeight / 2), behavior: "smooth" });
  }
}

function updateSourceControls() {
  document.querySelectorAll("[data-source]").forEach((button) => {
    const active = button.dataset.source === activeSource;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  document.getElementById("source-range").textContent = SOURCES[activeSource].rangeLabel;
}

function updateViewControls() {
  document.querySelectorAll("[data-view]").forEach((button) => button.classList.toggle("active", button.dataset.view === activeView));
}

function selectEvent(eventId, scroll) {
  const event = eventsById.get(eventId);
  if (!event) return;
  selectedEventId = eventId;
  activeSource = "fit";
  selectedLine = event.line;
  if (activeView === "mainline" && !inRanges(selectedLine, SOURCES.fit.mainRanges)) activeView = "full";
  updateSourceControls();
  updateViewControls();
  renderSource();
  updateTraceSelection(eventId);
  renderInspector("fit", selectedLine, eventId);
  updateSourceSelection(scroll);
  history.replaceState(null, "", `#L${selectedLine}`);
}

function selectLine(source, line, scroll) {
  if (!SOURCES[source] || line < SOURCES[source].start || line > SOURCES[source].end) return;
  activeSource = source;
  selectedLine = line;
  const exactEvents = source === "fit" ? runtimeEvents.filter((event) => event.line === line) : [];
  selectedEventId = exactEvents[0]?.id || null;
  if (activeView === "mainline" && !inRanges(line, SOURCES[source].mainRanges)) activeView = "full";
  updateSourceControls();
  updateViewControls();
  renderSource();
  updateTraceSelection(selectedEventId);
  renderInspector(source, line, selectedEventId);
  updateSourceSelection(scroll);
  const hash = source === "fit" ? `#L${line}` : `#helper-L${line}`;
  history.replaceState(null, "", hash);
}

function parseHash() {
  const helper = location.hash.match(/^#helper-L(\d+)$/);
  if (helper) return ["helper", Number(helper[1])];
  const fit = location.hash.match(/^#(?:fit-)?L(\d+)$/);
  if (fit) return ["fit", Number(fit[1])];
  return ["fit", 1036];
}

function wireControls() {
  document.querySelectorAll("[data-source]").forEach((button) => button.addEventListener("click", () => {
    const source = button.dataset.source;
    const line = source === "fit" ? 1036 : 85;
    selectLine(source, line, true);
  }));
  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
    activeView = button.dataset.view;
    updateViewControls();
    renderSource();
    updateSourceSelection(true);
  }));
  document.getElementById("line-jump").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    selectLine(activeSource, Number(event.currentTarget.value), true);
  });
  document.querySelectorAll("[data-jump-event]").forEach((button) => button.addEventListener("click", () => selectEvent(button.dataset.jumpEvent, true)));
  document.querySelectorAll("[data-open-source]").forEach((button) => button.addEventListener("click", () => selectLine(button.dataset.openSource, Number(button.dataset.line), true)));
}

async function init() {
  try {
    const entries = await Promise.all(Object.entries(SOURCES).map(async ([key, source]) => {
      const response = await fetch(source.file);
      if (!response.ok) throw new Error(`${source.file}: HTTP ${response.status}`);
      const text = (await response.text()).replace(/\n$/, "");
      return [key, text.split("\n")];
    }));
    sourceData = Object.fromEntries(entries);
    for (const [key, lines] of Object.entries(sourceData)) {
      const expected = SOURCES[key].end - SOURCES[key].start + 1;
      if (lines.length !== expected) throw new Error(`${key} 源码行数 ${lines.length}，预期 ${expected}`);
    }
    renderTrace();
    wireControls();
    updateSourceControls();
    updateViewControls();
    const [source, line] = parseHash();
    selectLine(source, line, true);
  } catch (error) {
    document.getElementById("source-code").innerHTML = `<div class="load-error">源码加载失败：${escapeHtml(error.message)}</div>`;
  }
}

init();
