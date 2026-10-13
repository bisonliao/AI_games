import os
import resource
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch


def make_run_id() -> str:
    now = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    return f'{now}_pid{os.getpid()}'


class EpisodeMetrics:
    def __init__(self, num_envs: int):
        self.returns = [0.0] * num_envs
        self.lengths = [0] * num_envs
        self.completed = 0
        self.self_draws = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.return_sum = 0.0
        self.length_sum = 0
        self.total_completed = 0
        self.total_self_draws = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.total_return_sum = 0.0
        self.total_length_sum = 0
        self.invalid_actions = 0

    def update(self, rewards, terminated, truncated, infos):
        for index, reward in enumerate(rewards):
            if infos[index].get('invalid_action'):
                self.invalid_actions += 1
            self.returns[index] += float(reward)
            self.lengths[index] += 1
            if bool(terminated[index]) or bool(truncated[index]):
                info = infos[index]
                reason = info.get('terminal_reason')
                episode_return = self.returns[index]
                episode_length = self.lengths[index]
                self.completed += 1
                self.return_sum += episode_return
                self.length_sum += episode_length
                self.total_completed += 1
                self.total_return_sum += episode_return
                self.total_length_sum += episode_length
                if reason == 'win':
                    self.wins += 1
                    self.self_draws += 1
                    self.total_wins += 1
                    self.total_self_draws += 1
                elif reason == 'draw':
                    self.draws += 1
                    self.total_draws += 1
                elif reason == 'loss':
                    self.losses += 1
                    self.total_losses += 1
                self.returns[index] = 0.0
                self.lengths[index] = 0

    def values(self):
        count = max(1, self.completed)
        total_count = max(1, self.total_completed)
        values = {
            # These are interval metrics, so a plateau means no episodes
            # completed during the interval rather than an old cumulative
            # average being repeated forever.
            'episode/self_draw_rate': self.self_draws / count,
            'episode/win_rate': self.wins / count,
            'episode/draw_rate': self.draws / count,
            'episode/loss_rate': self.losses / count,
            'episode/return': self.return_sum / count,
            'episode/length': self.length_sum / count,
            'episode/count': float(self.completed),
            'episode/total_count': float(self.total_completed),
            'episode/total_self_draw_rate': self.total_self_draws / total_count,
            'episode/invalid_action_count': float(self.invalid_actions),
        }
        self.completed = self.self_draws = self.wins = self.draws = self.losses = 0
        self.return_sum = self.length_sum = 0.0
        self.invalid_actions = 0
        return values


def resource_values():
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    gpu_memory_mb = -1.0
    gpu_utilization = -1.0
    if torch.cuda.is_available():
        gpu_memory_mb = float(torch.cuda.memory_allocated() / (1024 * 1024))
    return {
        'system/rss_mb': rss_mb,
        'system/gpu_memory_mb': gpu_memory_mb,
        'system/gpu_utilization': gpu_utilization,
    }


class MetricsLogger:
    def __init__(self, root: str = 'tb_logs', run_id: str | None = None):
        from torch.utils.tensorboard import SummaryWriter

        self.run_id = run_id or make_run_id()
        self.log_path = Path(root) / self.run_id
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_path), max_queue=128, flush_secs=30)
        self.last_time = time.monotonic()
        self.last_step = 0

    def log(self, step: int, values: dict[str, float]):
        for name, value in values.items():
            self.writer.add_scalar(name, float(value), step)
        self.writer.flush()
        self.last_step = step
        self.last_time = time.monotonic()

    def close(self):
        self.writer.close()
