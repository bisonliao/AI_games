import json
import os
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


class EpisodeRecorder:
    def __init__(self):
        self.rows = []

    def add(self, obs, action, reward, next_obs, terminated, truncated, info):
        self.rows.append((obs, action, reward, next_obs, terminated, truncated, info))

    @staticmethod
    def _mask(observation):
        if isinstance(observation, dict):
            return observation.get(
                'action_mask',
                np.ones(141, dtype=np.int8),
            )
        return np.ones(141, dtype=np.int8)

    @staticmethod
    def _vector(observation):
        if isinstance(observation, dict):
            return observation.get('vector', observation)
        return observation

    @staticmethod
    def _arrays(rows):
        vec = np.asarray(
            [EpisodeRecorder._vector(row[0]) for row in rows],
            dtype=np.float32,
        )
        nxt = np.asarray(
            [EpisodeRecorder._vector(row[3]) for row in rows],
            dtype=np.float32,
        )
        masks = (
            np.stack(
                [
                    EpisodeRecorder._mask(row[0])
                    for row in rows
                ]
            )
            if rows else np.zeros((0, 141), np.int8)
        )
        next_masks = (
            np.stack(
                [
                    EpisodeRecorder._mask(row[3])
                    for row in rows
                ]
            )
            if rows else np.zeros((0, 141), np.int8)
        )
        return {
            'observations': vec,
            'next_observations': nxt,
            'actions': np.asarray([r[1] for r in rows], np.int16),
            'rewards': np.asarray([r[2] for r in rows], np.float32),
            'terminated': np.asarray([r[4] for r in rows], bool),
            'truncated': np.asarray([r[5] for r in rows], bool),
            'action_mask': masks,
            'next_action_mask': next_masks,
        }

    def save(self, output):
        """Write one episode without reading or replacing historical data.

        Output is a directory. Legacy *.npz arguments use a sibling directory
        named after the file stem; the original archive is left untouched.
        """
        if not self.rows:
            return None
        directory = Path(output)
        if directory.suffix.lower() == '.npz':
            directory = directory.with_suffix('')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"episode_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%d_%H%M%S_%f}.npz"
        arrays = self._arrays(self.rows)
        metadata = {
            'count': len(self.rows),
            'format': 'mahjong-expert-v1',
            'action_count': 141,
            'observation': 'vector',
        }
        # Keep metadata in the same archive so the episode commits as one file.
        arrays['metadata'] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        fd, tmp_name = tempfile.mkstemp(prefix='.episode_', suffix='.tmp', dir=directory)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, 'wb') as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            # Atomic publication, with no overwrite even if a name collides.
            os.link(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path
