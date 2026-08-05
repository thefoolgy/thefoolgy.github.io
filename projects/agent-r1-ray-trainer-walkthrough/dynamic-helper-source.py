def filter_informative_prompt_groups(
    batch: DataProto,
    metric_name: str = "seq_reward",
    min_std: float = 1e-6,
) -> tuple[DataProto, list[object], dict[str, float]]:
    """Keep prompt groups whose complete trajectory returns are not all equal.

    Agent-R1 expands one rollout into multiple step rows. Dynamic sampling must
    therefore sum row-level rewards by ``trajectory_uids`` before measuring the
    variation among rollouts sharing the same prompt ``uid``.
    """
    if metric_name == "seq_reward":
        row_metrics = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
    elif metric_name == "seq_final_reward":
        row_metrics = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
    elif metric_name in batch.non_tensor_batch:
        row_metrics = np.asarray(batch.non_tensor_batch[metric_name], dtype=np.float64)
    else:
        raise KeyError(
            f"Dynamic-sampling metric {metric_name!r} is unavailable. "
            "Use 'seq_reward', 'seq_final_reward', or a non-tensor batch metric."
        )

    prompt_uids = batch.non_tensor_batch.get("uid")
    trajectory_uids = batch.non_tensor_batch.get("trajectory_uids")
    if prompt_uids is None or trajectory_uids is None:
        raise KeyError("Dynamic sampling requires both 'uid' and 'trajectory_uids' in the AgentFlow batch.")
    if not (len(batch) == len(prompt_uids) == len(trajectory_uids) == len(row_metrics)):
        raise ValueError("Dynamic-sampling row fields are not aligned.")

    trajectory_totals: dict[object, float] = {}
    trajectory_prompt: dict[object, object] = {}
    trajectory_order: list[object] = []
    prompt_order: list[object] = []
    seen_prompts: set[object] = set()

    for prompt_uid, trajectory_uid, row_metric in zip(prompt_uids, trajectory_uids, row_metrics, strict=True):
        if prompt_uid not in seen_prompts:
            seen_prompts.add(prompt_uid)
            prompt_order.append(prompt_uid)
        if trajectory_uid not in trajectory_totals:
            trajectory_totals[trajectory_uid] = 0.0
            trajectory_prompt[trajectory_uid] = prompt_uid
            trajectory_order.append(trajectory_uid)
        elif trajectory_prompt[trajectory_uid] != prompt_uid:
            raise ValueError(f"Trajectory {trajectory_uid!r} is associated with multiple prompt uids.")
        trajectory_totals[trajectory_uid] += float(row_metric)

    prompt_returns: dict[object, list[float]] = defaultdict(list)
    for trajectory_uid in trajectory_order:
        prompt_returns[trajectory_prompt[trajectory_uid]].append(trajectory_totals[trajectory_uid])

    kept_prompt_uids = [
        prompt_uid
        for prompt_uid in prompt_order
        if len(prompt_returns[prompt_uid]) == 1 or np.std(prompt_returns[prompt_uid]) > min_std
    ]
    kept_set = set(kept_prompt_uids)
    kept_row_indices = [idx for idx, prompt_uid in enumerate(prompt_uids) if prompt_uid in kept_set]
    filtered = batch.select_idxs(kept_row_indices)

    num_prompts = len(prompt_order)
    num_kept = len(kept_prompt_uids)
    stats = {
        "generated_prompts": float(num_prompts),
        "kept_prompts": float(num_kept),
        "rejected_prompts": float(num_prompts - num_kept),
        "generated_trajectories": float(len(trajectory_order)),
        "kept_rows": float(len(kept_row_indices)),
    }
    return filtered, kept_prompt_uids, stats


def select_prompt_groups(batch: DataProto, prompt_uids: Sequence[object]) -> DataProto:
    """Select complete AgentFlow step rows for an ordered list of prompt groups."""
    selected = set(prompt_uids)
    indices = [idx for idx, prompt_uid in enumerate(batch.non_tensor_batch["uid"]) if prompt_uid in selected]
    return batch.select_idxs(indices)
