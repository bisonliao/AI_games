import tempfile
import unittest
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from MahjongEnv.dataset import EpisodeRecorder
from play import run


class EpisodeStorageTests(unittest.TestCase):
    def recorder(self):
        rec = EpisodeRecorder()
        obs = {'vector': np.zeros(178, np.float32), 'action_mask': np.ones(141, np.int8)}
        rec.add(obs, 0, 1.0, obs, True, False, {})
        return rec

    def test_separate_runs_keep_existing_files_without_loading_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / 'expert_data.npz'
            legacy.write_bytes(b'old archive must remain untouched')
            args = Namespace(headless=True, seed=1, episodes=2, output=str(legacy))
            with patch('numpy.load', side_effect=AssertionError('history was loaded')):
                run(args)
                first = {p: p.read_bytes() for p in legacy.with_suffix('').glob('*.npz')}
                self.assertEqual(len(first), 2)
                run(args)
            files = list(legacy.with_suffix('').glob('*.npz'))
            self.assertEqual(len(files), 4)
            for p, content in first.items():
                self.assertEqual(p.read_bytes(), content)
            self.assertEqual(legacy.read_bytes(), b'old archive must remain untouched')
            for p in files:
                with np.load(p, allow_pickle=False) as data:
                    self.assertEqual(int(data['terminated'].sum()), 1)
                    self.assertTrue(data['terminated'][-1])
                    self.assertEqual(data['next_action_mask'].shape[1], 141)

    def test_failed_write_publishes_no_partial_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self.recorder().save(tmp)
            content = saved.read_bytes()
            with patch('numpy.savez_compressed', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    self.recorder().save(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [saved])
            self.assertEqual(saved.read_bytes(), content)

    def test_collision_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('MahjongEnv.dataset.datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 12, 15, 30, 45, 123456)
                saved = self.recorder().save(tmp)
                content = saved.read_bytes()
                self.assertEqual(saved.name, 'episode_20261012_153045_123456.npz')
                with self.assertRaises(FileExistsError):
                    self.recorder().save(tmp)
            self.assertEqual(saved.read_bytes(), content)

    def test_raw_vector_observations_get_default_masks(self):
        recorder = EpisodeRecorder()
        observation = np.zeros(306, dtype=np.float32)
        recorder.add(observation, 0, 0.0, observation, True, False, {})
        arrays = recorder._arrays(recorder.rows)
        self.assertEqual(arrays['action_mask'].shape, (1, 141))
        self.assertEqual(arrays['next_action_mask'].shape, (1, 141))


if __name__ == '__main__':
    unittest.main()
