import argparse
import contextlib
import io
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from robot_arm.octo.train import StageBatchSampler, learning_rate_factor, resolve_settings


class TrainingOptionsTests(unittest.TestCase):
    def settings(self, resume=None, stored=None, **overrides):
        values = dict(batch_size=None, accumulation=None, lr=None, warmup=None,
                      lr_decay=None, decay_start=None, decay_end=None,
                      min_lr_ratio=None, sampling=None, resume=resume)
        values.update(overrides)
        return resolve_settings(SimpleNamespace(**values), stored or {}, argparse.ArgumentParser())

    def test_defaults_boundaries_and_disabled_decay(self):
        settings = self.settings()
        self.assertEqual(settings["sampling"], "balanced")
        for update, expected in [(1, 1 / 500), (500, 1), (5000, 1),
                                 (7500, 0.55), (10000, 0.1), (20000, 0.1)]:
            self.assertAlmostEqual(learning_rate_factor(update - 1, settings), expected)
        disabled = self.settings(lr_decay=False)
        self.assertEqual(learning_rate_factor(9999, disabled), 1)

    def test_legacy_resume_and_config_mismatch(self):
        legacy = dict(settings=dict(batch_size=4, accumulation=8, lr=3e-5, warmup=500))
        settings = self.settings(resume="legacy.pt", stored=legacy)
        self.assertFalse(settings["lr_decay"])
        self.assertEqual(settings["sampling"], "balanced")
        for override in [dict(lr_decay=True), dict(sampling="random")]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.settings(resume="legacy.pt", stored=legacy, **override)
        saved = self.settings(sampling="random", decay_end=12000)
        self.assertEqual(self.settings(resume="new.pt", stored=dict(settings=saved)), saved)

    def test_invalid_schedule(self):
        for override in [dict(decay_start=100), dict(decay_end=5000), dict(min_lr_ratio=0),
                         dict(min_lr_ratio=1.1), dict(lr=float("nan"))]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.settings(**override)

    def test_scheduler_resume_preserves_actual_lr(self):
        settings = self.settings(warmup=2, decay_start=4, decay_end=8)
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=3e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(step, settings))
        for _ in range(6):
            optimizer.step()
            scheduler.step()
        restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=3e-5)
        restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: learning_rate_factor(step, settings))
        restored_optimizer.load_state_dict(optimizer.state_dict())
        restored_scheduler.load_state_dict(scheduler.state_dict())
        for _ in range(5):
            self.assertEqual(optimizer.param_groups[0]["lr"], restored_optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            restored_optimizer.step()
            restored_scheduler.step()

    def test_sampling_distributions_and_rng_resume(self):
        stages = [0] * 80 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5
        dataset = SimpleNamespace(indices=[("episode", index, stage) for index, stage in enumerate(stages)])
        for mode, expected in [("balanced", [0.2] * 5), ("random", [0.8, 0.05, 0.05, 0.05, 0.05])]:
            generator = torch.Generator().manual_seed(42)
            sampler = StageBatchSampler(dataset, 20000, 1, generator, sampling=mode)
            state = generator.get_state()
            indices = next(iter(sampler))
            frequencies = np.bincount(np.array(stages)[indices], minlength=5) / len(indices)
            np.testing.assert_allclose(frequencies, expected, atol=0.015)
            generator.set_state(state)
            self.assertEqual(indices, next(iter(StageBatchSampler(dataset, 20000, 1, generator, sampling=mode))))


if __name__ == "__main__":
    unittest.main()
