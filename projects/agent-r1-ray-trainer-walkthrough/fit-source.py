    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=make_json_safe(val_metrics), step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        filter_groups_config = self.config.algorithm.get("filter_groups", None)
        dynamic_sampling_enabled = bool(filter_groups_config and filter_groups_config.get("enable", False))
        timing_raw = defaultdict(float)
        accumulated_batch = None
        accumulated_prompt_uids: list[object] = []
        num_gen_batches = 0
        dynamic_sampling_totals: dict[str, float] = defaultdict(float)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1

                # add uid to batch
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(new_batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        for key, value in gen_batch_output.meta_info["timing"].items():
                            timing_raw[key] += value
                        gen_batch_output.meta_info.pop("timing", None)

                    if AgentAdvantageEstimator(self.config.algorithm.adv_estimator) == AgentAdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")
                        # TODO: implement REMAX advantage estimation for agent flow.
                        raise NotImplementedError("REMAX advantage estimation is not supported for agent flow.")

                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    num_steps = gen_batch_output.meta_info.pop("num_steps")
                    new_batch = new_batch.sample_level_repeat(num_steps)
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                            reward_tensor = self.reward_loop_manager.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=new_batch, config=self.config, tokenizer=self.tokenizer
                            )
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                new_batch, reward_fn=self.reward_fn, return_dict=False
                            )
                        new_batch.batch["token_level_scores"] = reward_tensor
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {key: np.asarray(values) for key, values in reward_extra_infos_dict.items()}
                            )

                    if dynamic_sampling_enabled:
                        metric_name = filter_groups_config.get("metric", "seq_reward")
                        filtered_batch, kept_prompt_uids, filter_stats = filter_informative_prompt_groups(
                            new_batch, metric_name=metric_name
                        )
                        for key, value in filter_stats.items():
                            dynamic_sampling_totals[key] += value

                        if kept_prompt_uids:
                            accumulated_batch = (
                                filtered_batch
                                if accumulated_batch is None
                                else DataProto.concat([accumulated_batch, filtered_batch])
                            )
                            accumulated_prompt_uids.extend(kept_prompt_uids)

                        prompt_batch_size = self.config.data.train_batch_size
                        if len(accumulated_prompt_uids) < prompt_batch_size:
                            max_num_gen_batches = filter_groups_config.get("max_num_gen_batches", 0)
                            print(
                                "Dynamic sampling refill: "
                                f"kept={len(accumulated_prompt_uids)}/{prompt_batch_size}, "
                                f"generation_batches={num_gen_batches}/{max_num_gen_batches or 'unlimited'}"
                            )
                            if max_num_gen_batches > 0 and num_gen_batches >= max_num_gen_batches:
                                raise ValueError(
                                    "Dynamic sampling could not fill one optimizer batch: "
                                    f"kept {len(accumulated_prompt_uids)} informative prompts after "
                                    f"{num_gen_batches} generation batches."
                                )
                            continue

                        selected_prompt_uids = accumulated_prompt_uids[:prompt_batch_size]
                        batch = select_prompt_groups(accumulated_batch, selected_prompt_uids)
                    else:
                        batch = new_batch

                    # From here onward only the complete optimizer batch is consumed.
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()
                    batch = self._pad_dataproto_to_world_size(batch)

                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=AgentAdvantageEstimator(self.config.algorithm.adv_estimator),
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    reward_extra_infos_dict = {
                        key: make_json_safe(batch.non_tensor_batch[key])
                        for key in reward_extra_infos_dict
                        if key in batch.non_tensor_batch
                    }

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # Temporarily replace response_mask for critic
                            response_mask = batch.batch["response_mask"]
                            adv_estimator = AgentAdvantageEstimator(self.config.algorithm.adv_estimator)
                            value_mask = critic_vf_loss_response_mask(response_mask, adv_estimator)
                            sample_mask = batch.batch.get("sample_mask", None)
                            if sample_mask is not None:
                                value_mask[~sample_mask.to(dtype=torch.bool)] = 0
                            batch.batch["response_mask"] = value_mask

                            # update critic
                            critic_output = self._update_critic(batch)

                            # restore response_mask
                            batch.batch["response_mask"] = response_mask
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                valid_batch, _ = get_valid_data(batch)

                metrics.update(compute_data_metrics(batch=valid_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=valid_batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=valid_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                metrics["dynamic_sampling/enabled"] = float(dynamic_sampling_enabled)
                metrics["dynamic_sampling/gen_batches"] = float(num_gen_batches)
                if dynamic_sampling_enabled:
                    for key, value in dynamic_sampling_totals.items():
                        metrics[f"dynamic_sampling/{key}"] = value
                    generated_prompts = dynamic_sampling_totals["generated_prompts"]
                    metrics["dynamic_sampling/acceptance_rate"] = (
                        dynamic_sampling_totals["kept_prompts"] / generated_prompts if generated_prompts else 0.0
                    )
                    metrics["dynamic_sampling/selected_prompts"] = float(self.config.data.train_batch_size)

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=make_json_safe(metrics), step=self.global_steps)

                timing_raw = defaultdict(float)
                accumulated_batch = None
                accumulated_prompt_uids = []
                num_gen_batches = 0
                dynamic_sampling_totals = defaultdict(float)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
