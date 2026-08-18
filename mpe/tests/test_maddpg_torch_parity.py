import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

from experiments.train_torch import (
    _checkpoint_metadata,
    _checkpoint_payload,
    _load_trainers_from_checkpoint,
    _transition_flags,
)
from maddpg.common.distributions_torch import ActionSpec
from maddpg.trainer.maddpg_torch import (
    MADDPGAgentTrainer,
    actor_loss,
    bellman_target,
    clip_grad_norm_per_parameter,
)


def _args(target_init="copy"):
    return SimpleNamespace(
        env_backend="legacy",
        scenario="simple",
        policy_mode="official",
        target_init=target_init,
        num_units=16,
        lr=1e-2,
        batch_size=4,
        max_episode_len=1,
        gamma=0.95,
    )


def _trainer(target_init="copy"):
    return MADDPGAgentTrainer(
        "agent_0",
        None,
        [(4,)],
        [spaces.Discrete(5)],
        0,
        _args(target_init),
        device=torch.device("cpu"),
        action_spec_n=[ActionSpec("official", (5,))],
    )


class TorchTrainerParityTest(unittest.TestCase):
    def test_target_copy_default_and_independent_ablation(self):
        copied = _trainer("copy")
        for online, target in zip(
            copied.p_net.parameters(), copied.target_p_net.parameters()
        ):
            torch.testing.assert_close(online, target)

        independent = _trainer("independent")
        differences = [
            not torch.equal(online, target)
            for online, target in zip(
                independent.p_net.parameters(), independent.target_p_net.parameters()
            )
        ]
        self.assertTrue(any(differences))

    def test_linear_biases_are_zero(self):
        trainer = _trainer()
        for module in trainer.p_net.modules():
            if isinstance(module, torch.nn.Linear):
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))

    def test_gradient_clipping_is_per_parameter(self):
        p1 = torch.nn.Parameter(torch.zeros(2))
        p2 = torch.nn.Parameter(torch.zeros(2))
        p1.grad = torch.tensor([3.0, 4.0])
        p2.grad = torch.tensor([0.0, 2.0])
        pre_clip = clip_grad_norm_per_parameter([p1, p2], 0.5)

        torch.testing.assert_close(pre_clip, torch.sqrt(torch.tensor(29.0)))
        torch.testing.assert_close(torch.linalg.vector_norm(p1.grad), torch.tensor(0.5))
        torch.testing.assert_close(torch.linalg.vector_norm(p2.grad), torch.tensor(0.5))

    def test_truncation_ends_episode_but_does_not_set_replay_done(self):
        terminated, done, terminal, ended = _transition_flags(
            {"agent_0": False},
            {"agent_0": True},
            ["agent_0"],
            episode_step=25,
            max_steps=25,
        )
        self.assertEqual(terminated, [False])
        self.assertTrue(done)
        self.assertTrue(terminal)
        self.assertTrue(ended)

        rewards = torch.tensor([2.0, 2.0])
        replay_dones = torch.tensor([0.0, 1.0])
        next_q = torch.tensor([3.0, 3.0])
        target = bellman_target(rewards, replay_dones, next_q, gamma=0.5)
        torch.testing.assert_close(target, torch.tensor([3.5, 2.0]))

    def test_actor_loss_matches_official_formula(self):
        q_values = torch.tensor([1.0, -3.0])
        logits = torch.tensor([[1.0, -1.0], [2.0, 0.0]])

        total, policy_gradient, regularization = actor_loss(q_values, logits)

        torch.testing.assert_close(policy_gradient, torch.tensor(1.0))
        torch.testing.assert_close(regularization, torch.tensor(1.5))
        torch.testing.assert_close(total, torch.tensor(1.0015))

    def test_polyak_update_matches_tf_expression(self):
        trainer = _trainer()
        with torch.no_grad():
            for parameter in trainer.p_net.parameters():
                parameter.fill_(2.0)
            for parameter in trainer.target_p_net.parameters():
                parameter.fill_(4.0)

        trainer._soft_update(trainer.p_net, trainer.target_p_net)

        for parameter in trainer.target_p_net.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 3.98))

    def test_update_is_finite_and_uses_soft_action_dimension(self):
        trainer = _trainer()
        for index in range(4):
            obs = np.full(4, index / 10.0, dtype=np.float32)
            action = np.full(5, 0.2, dtype=np.float32)
            trainer.experience(obs, action, 1.0, obs + 0.1, False, False)
        result = trainer.update([trainer], 100)

        self.assertIsNotNone(result)
        for value in result.values():
            self.assertTrue(np.isfinite(value))

    def test_checkpoint_restores_networks_and_adam_moments(self):
        source = _trainer()
        source.p_optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in source.p_net.parameters()).backward()
        source.p_optimizer.step()
        source.q_optimizer.zero_grad()
        sum(parameter.square().sum() for parameter in source.q_net.parameters()).backward()
        source.q_optimizer.step()

        state = copy.deepcopy(source.checkpoint_state())
        restored = _trainer()
        restored.load_checkpoint_state(state, load_optimizers=True)

        for source_parameter, restored_parameter in zip(
            source.p_net.parameters(), restored.p_net.parameters()
        ):
            torch.testing.assert_close(source_parameter, restored_parameter)
        self.assertEqual(
            source.p_optimizer.state_dict()["state"].keys(),
            restored.p_optimizer.state_dict()["state"].keys(),
        )
        for key, value in source.p_optimizer.state_dict()["state"].items():
            for state_name, state_value in value.items():
                restored_value = restored.p_optimizer.state_dict()["state"][key][state_name]
                if torch.is_tensor(state_value):
                    torch.testing.assert_close(state_value, restored_value)
                else:
                    self.assertEqual(state_value, restored_value)

    def test_new_checkpoint_envelope_restores_progress_strictly(self):
        args = _args()
        spec = [ActionSpec("official", (5,))]
        source = _trainer()
        payload = _checkpoint_payload(args, spec, [source], 1234, 56)
        restored = _trainer()

        train_step, episodes = _load_trainers_from_checkpoint(
            payload,
            [restored],
            _checkpoint_metadata(args, spec),
            load_optimizers=True,
        )

        self.assertEqual((train_step, episodes), (1234, 56))
        for source_parameter, restored_parameter in zip(
            source.target_q_net.parameters(), restored.target_q_net.parameters()
        ):
            torch.testing.assert_close(source_parameter, restored_parameter)

        version_2_payload = copy.deepcopy(payload)
        version_2_payload["checkpoint_version"] = 2
        version_2_payload["metadata"] = {
            key: payload["metadata"][key]
            for key in (
                "env_backend",
                "scenario",
                "policy_mode",
                "target_init",
                "action_specs",
            )
        }
        train_step, episodes = _load_trainers_from_checkpoint(
            version_2_payload,
            [restored],
            _checkpoint_metadata(args, spec),
            load_optimizers=True,
        )
        self.assertEqual((train_step, episodes), (1234, 56))

        version_3_payload = copy.deepcopy(payload)
        version_3_payload["checkpoint_version"] = 3
        del version_3_payload["metadata"]["algorithm"]
        train_step, episodes = _load_trainers_from_checkpoint(
            version_3_payload,
            [restored],
            _checkpoint_metadata(args, spec),
            load_optimizers=True,
        )
        self.assertEqual((train_step, episodes), (1234, 56))

    def test_old_or_mismatched_checkpoints_are_rejected(self):
        args = _args()
        spec = [ActionSpec("official", (5,))]
        trainer = _trainer()
        metadata = _checkpoint_metadata(args, spec)

        with self.assertRaisesRegex(ValueError, "unsupported checkpoint"):
            _load_trainers_from_checkpoint(
                {"p_nets": []}, [trainer], metadata, load_optimizers=False
            )

        payload = _checkpoint_payload(args, spec, [trainer], 1, 1)
        mismatched_metadata = dict(metadata, scenario="simple_spread")
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            _load_trainers_from_checkpoint(
                payload,
                [trainer],
                mismatched_metadata,
                load_optimizers=False,
            )

        incomplete = copy.deepcopy(payload)
        del incomplete["trainers"][0]["p_optimizer"]
        with self.assertRaisesRegex(ValueError, "p_optimizer"):
            _load_trainers_from_checkpoint(
                incomplete, [trainer], metadata, load_optimizers=False
            )


if __name__ == "__main__":
    unittest.main()
