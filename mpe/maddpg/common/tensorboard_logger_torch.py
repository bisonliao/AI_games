"""Interval aggregation for PyTorch MADDPG TensorBoard metrics."""

from collections import defaultdict, deque

from torch.utils.tensorboard import SummaryWriter


class _RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0

    def add(self, value):
        value = float(value)
        self.count += 1
        self.total += value

    @property
    def mean(self):
        return self.total / self.count

class TensorBoardIntervalLogger:
    """Aggregate every observed metric and periodically write interval means.

    ``writer`` is injectable for tests. Supplying neither ``writer`` nor
    ``log_dir`` creates a disabled no-op logger.
    """

    UPDATE_METRICS = {
        "loss/q_loss": "q_loss",
        "loss/p_loss": "p_loss",
        "loss/pg_loss": "pg_loss",
        "loss/p_reg": "p_reg",
        "q/mean_q": "mean_q",
        "q/mean_target_q": "mean_target_q",
        "q/mean_target_q_next": "mean_target_q_next",
        "q/std_target_q": "std_target_q",
        "reward/step_batch_reward": "mean_batch_rew",
        "policy/action_entropy": "action_entropy",
        "policy/action_std_mean": "action_std_mean",
        "grad/q_grad_norm": "q_grad_norm",
        "grad/p_grad_norm": "p_grad_norm",
    }

    def __init__(
        self,
        interval,
        initial_step=0,
        log_dir=None,
        writer=None,
    ):
        if interval < 0:
            raise ValueError("TensorBoard interval must be non-negative")
        self.interval = int(interval)
        self.last_log_step = int(initial_step)
        self.writer = writer
        if self.writer is None and log_dir is not None and interval > 0:
            self.writer = SummaryWriter(log_dir=log_dir)
        self._stats = defaultdict(_RunningStats)
        self._latest = {}
        self._success_total = 0.0
        self._success_count = 0
        self._recent_successes = deque(maxlen=100)
        self._recent_episode_rewards = deque(maxlen=10)

    @property
    def enabled(self):
        return self.writer is not None and self.interval > 0

    def record_training_update(self, update_results):
        """Record one optimizer event, averaging agents before time."""

        if not self.enabled or not update_results:
            return
        for tag, result_key in self.UPDATE_METRICS.items():
            values = [float(result[result_key]) for result in update_results]
            self._stats[tag].add(sum(values) / len(values))
        if len(update_results) > 1:
            for agent_index, result in enumerate(update_results):
                self._stats[
                    "loss/q_loss_agent{}".format(agent_index)
                ].add(result["q_loss"])
                self._stats[
                    "policy/action_entropy_agent{}".format(agent_index)
                ].add(result["action_entropy"])

    def record_episode(
        self,
        episode_reward,
        episode_length,
        agent_episode_rewards,
        task_metrics,
    ):
        """Record every completed episode without dropping intermediate data."""

        if not self.enabled:
            return
        reward = float(episode_reward)
        self._stats["reward/episode_reward"].add(reward)
        self._stats["env/episode_length"].add(episode_length)
        self._recent_episode_rewards.append(reward)
        if agent_episode_rewards is not None:
            for agent_index, agent_reward in enumerate(agent_episode_rewards):
                self._stats[
                    "reward/agent{}_episode_reward".format(agent_index)
                ].add(agent_reward)
        for metric_name, metric_value in task_metrics.items():
            self._stats["task/{}".format(metric_name)].add(metric_value)

        if "episode_success" in task_metrics:
            success = float(task_metrics["episode_success"])
            self._success_total += success
            self._success_count += 1
            self._recent_successes.append(success)

    def record_scalar(self, tag, value):
        """Include a generic scalar in the next interval mean."""

        if self.enabled:
            self._stats[tag].add(value)

    def record_latest(self, tag, value):
        """Write the most recently supplied gauge at the next flush."""

        if self.enabled:
            self._latest[tag] = float(value)

    def _build_interval_metrics(self):
        metrics = {
            tag: stats.mean for tag, stats in self._stats.items()
        }
        reward_stats = self._stats.get("reward/episode_reward")
        if reward_stats is not None and reward_stats.count:
            metrics["reward/episode_reward_roll10"] = sum(
                self._recent_episode_rewards
            ) / len(self._recent_episode_rewards)
        if self._success_count:
            metrics["task/success_rate"] = (
                self._success_total / self._success_count
            )
            metrics["task/success_rate_roll100"] = sum(
                self._recent_successes
            ) / len(self._recent_successes)
        metrics.update(self._latest)
        return metrics

    def maybe_flush(self, train_step, force=False):
        if not self.enabled or (not self._stats and not self._latest):
            return False
        train_step = int(train_step)
        if not force and train_step - self.last_log_step < self.interval:
            return False
        for tag, value in self._build_interval_metrics().items():
            self.writer.add_scalar(tag, value, train_step)
        self._stats.clear()
        self._latest.clear()
        self.last_log_step = train_step
        return True

    def write_immediate(self, metrics, train_step, flush=False):
        """Write checkpoint evaluation metrics without interval aggregation."""

        if not self.enabled:
            return
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, int(train_step))
        if flush:
            self.writer.flush()

    def close(self, train_step):
        if self.writer is None:
            return
        self.maybe_flush(train_step, force=True)
        self.writer.flush()
        self.writer.close()
