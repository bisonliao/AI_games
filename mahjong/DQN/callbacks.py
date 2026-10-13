import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class ActionStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.invalid = 0
        self.mask_counts = []
        self.action_counts = np.zeros(6, dtype=np.int64)

    def _on_step(self):
        infos = self.locals.get('infos', [])
        new_obs = self.locals.get('new_obs')
        masks = new_obs.get('action_mask') if isinstance(new_obs, dict) else None
        self.invalid += sum(bool(info.get('invalid_action')) for info in infos)
        actions = np.asarray(self.locals.get('actions', []), dtype=np.int64).reshape(-1)
        for action in actions:
            self.action_counts[min(int(action) // 28, 5)] += 1
        if masks is not None:
            self.mask_counts.extend(np.asarray(masks).sum(axis=1).tolist())
        self.logger.record('episode/invalid_action_count', self.invalid)
        if self.mask_counts:
            self.logger.record('train/legal_actions_mean', float(np.mean(self.mask_counts[-1000:])))
        names = ('discard', 'peng', 'ming_kong', 'concealed_kong', 'added_kong', 'pass')
        total_actions = max(int(self.action_counts.sum()), 1)
        for index, name in enumerate(names):
            self.logger.record(f'train/action_rate/{name}', self.action_counts[index] / total_actions)
        return True

def make_eval_callback(eval_env, eval_freq, n_eval_episodes, best_model_save_path, log_path):
    return ReasonEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=max(1, eval_freq),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )


class ReasonEvalCallback(EvalCallback):
    """EvalCallback variant that reports Mahjong terminal outcomes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reason_counts = {'win': 0, 'draw': 0, 'loss': 0}

    def _log_success_callback(self, locals_, globals_):
        info = locals_.get('info', {})
        reason = info.get('terminal_reason')
        if reason in self._reason_counts:
            self._reason_counts[reason] += 1
        return super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._reason_counts = {'win': 0, 'draw': 0, 'loss': 0}
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            total = max(sum(self._reason_counts.values()), 1)
            self.logger.record('eval/self_draw_rate', self._reason_counts['win'] / total)
            self.logger.record('eval/win_rate', self._reason_counts['win'] / total)
            self.logger.record('eval/draw_rate', self._reason_counts['draw'] / total)
            self.logger.record('eval/loss_rate', self._reason_counts['loss'] / total)
            self.logger.dump(self.num_timesteps)
        return continue_training
