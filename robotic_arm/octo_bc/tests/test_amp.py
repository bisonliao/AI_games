import unittest

import torch

from robot_arm.octo.train import discard_nonfinite_update, prepare_gradients, StageBatchSampler


class AMPTests(unittest.TestCase):
    def test_overflow_skips_and_next_update_recovers(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.amp.GradScaler("cpu", init_scale=16, growth_interval=2)
        original = parameter.detach().clone()
        scaler.scale(parameter.square().sum()).backward()
        parameter.grad.fill_(float("inf"))
        self.assertIsNone(prepare_gradients(optimizer, scaler, [parameter]))
        torch.testing.assert_close(parameter, original)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(scheduler.last_epoch, 0)
        self.assertEqual(scaler.get_scale(), 8)
        self.assertEqual(scaler.state_dict()["_growth_tracker"], 0)
        self.assertIsNone(parameter.grad)
        scaler.scale(parameter.square().sum()).backward()
        norm = prepare_gradients(optimizer, scaler, [parameter])
        self.assertAlmostEqual(float(norm), 2)
        self.assertLessEqual(float(parameter.grad.abs().max()), 1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertLess(float(parameter.detach()), 1)
        self.assertEqual(scheduler.last_epoch, 1)

    def test_bad_forward_discards_entire_accumulation(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cpu", init_scale=16)
        discard_nonfinite_update(optimizer, scaler, "loss")
        self.assertEqual(scaler.get_scale(), 8)
        scaler.scale(parameter.square().sum() / 8).backward()
        discard_nonfinite_update(optimizer, scaler, "loss")
        self.assertIsNone(parameter.grad)
        self.assertEqual(scaler.get_scale(), 4)
        scaler.scale(parameter.square().sum()).backward()
        self.assertIsNotNone(prepare_gradients(optimizer, scaler, [parameter]))
        scaler.step(optimizer)
        scaler.update()

    def test_nan_and_finite_gradient_norm_overflow(self):
        for value in [float("nan"), 3e38]:
            parameter = torch.nn.Parameter(torch.ones(2))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            scaler = torch.amp.GradScaler("cpu", init_scale=1)
            scaler.scale(parameter.sum()).backward()
            parameter.grad.fill_(value)
            self.assertIsNone(prepare_gradients(optimizer, scaler, [parameter]))
            torch.testing.assert_close(parameter, torch.ones(2))
            self.assertEqual(scaler.get_scale(), 0.5)

    def test_fp32_nonfinite_gradient_still_fails(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        parameter.grad = torch.tensor([float("inf")])
        with self.assertRaisesRegex(FloatingPointError, "FP32"):
            prepare_gradients(optimizer, scaler, [parameter])
        torch.testing.assert_close(parameter, torch.tensor([1.0]))

    def test_sampler_allows_extra_attempts(self):
        from types import SimpleNamespace
        dataset = SimpleNamespace(indices=[("episode", stage, stage) for stage in range(5)])
        sampler = StageBatchSampler(dataset, 4, None, torch.Generator().manual_seed(42))
        iterator = iter(sampler)
        for _ in range(20):
            self.assertEqual(len(next(iterator)), 4)


if __name__ == "__main__":
    unittest.main()
