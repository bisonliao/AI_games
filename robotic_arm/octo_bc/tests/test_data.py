import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from robot_arm.octo.data import (validate_episode, make_manifest, action_statistics,
                                ExpertDataset, normalize_action, denormalize_action)
from robot_arm.octo.train import StageBatchSampler


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        for episode in range(12):
            self.write_episode(episode, success=episode < 10)

    def tearDown(self):
        self.temporary.cleanup()

    def write_episode(self, number, success=True, broken=False):
        image = np.zeros((5, 18, 96, 96), dtype=np.uint8)
        for step in range(5):
            for view in range(3):
                image[step, view * 3:view * 3 + 3] = max(0, step - 1) * 10 + view
                image[step, 9 + view * 3:12 + view * 3] = step * 10 + view
        if broken:
            image[2, 0] = 255
        action = np.linspace(-1, 1, 20, dtype=np.float32).reshape(5, 4)
        proprio = np.arange(130, dtype=np.float32).reshape(5, 26)
        proprio[0, 22:26] = 0
        proprio[1:, 22:26] = action[:-1]
        np.savez_compressed(self.directory / f"episode_{number:06}.npz", format_version=2,
                            image=image, proprio=proprio,
                            action=action, reward=np.zeros(5), stage=np.arange(5, dtype=np.int8),
                            terminated=np.array([False] * 4 + [True]), truncated=np.zeros(5, dtype=bool),
                            success=success, episode_metadata=json.dumps({"seed": number}))

    def test_split_and_statistics(self):
        manifest = make_manifest(self.directory)
        self.assertEqual(len(manifest["train"]), 9)
        self.assertEqual(len(manifest["validation"]), 1)
        self.assertEqual(len(manifest["diagnostic"]), 2)
        self.assertFalse(set(manifest["train"]) & set(manifest["validation"]))
        self.assertEqual(manifest, make_manifest(self.directory))
        statistics = action_statistics(self.directory, manifest["train"])
        action = np.array([0.1, -0.1, 0.5, -0.5], dtype=np.float32)
        np.testing.assert_allclose(denormalize_action(normalize_action(action, statistics), statistics), action, atol=1e-6)

    def test_three_views_history_and_masks(self):
        manifest = make_manifest(self.directory)
        dataset = ExpertDataset(self.directory, manifest, "train", action_statistics(self.directory, manifest["train"]))
        item = dataset[2]
        self.assertEqual(item["observation"]["image_xy"].shape, (2, 3, 256, 256))
        np.testing.assert_allclose(item["observation"]["image_yz"][:, 0, 0, 0], [12, 22], atol=1e-6)
        np.testing.assert_allclose(item["observation"]["proprio"][:, 0], [26, 52])
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch["observation"]["timestep_pad_mask"].shape, (2, 2))
        self.assertTrue(batch["observation"]["pad_mask_dict"]["image_xy"].all())
        first = dataset[5]
        torch.testing.assert_close(first["observation"]["proprio"][0], first["observation"]["proprio"][1])

    def test_balanced_sampler_and_resume(self):
        manifest = make_manifest(self.directory)
        dataset = ExpertDataset(self.directory, manifest, "train", action_statistics(self.directory, manifest["train"]))
        generator = torch.Generator().manual_seed(42)
        iterator = iter(StageBatchSampler(dataset, 4, 3, generator))
        next(iterator)
        state = generator.get_state()
        expected = next(iterator)
        generator.set_state(state)
        self.assertEqual(expected, next(iter(StageBatchSampler(dataset, 4, 1, generator))))

    def test_history_corruption_fails_with_filename(self):
        self.write_episode(0, broken=True)
        with self.assertRaisesRegex(ValueError, "episode_000000"):
            validate_episode(self.directory / "episode_000000.npz")


if __name__ == "__main__":
    unittest.main()
