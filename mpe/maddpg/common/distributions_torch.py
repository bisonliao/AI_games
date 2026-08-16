"""PyTorch probability distributions used by the MADDPG policies.

The original TensorFlow implementation represents MPE ``Discrete`` actions as
soft Gumbel-Softmax vectors. Old MPE consumes those vectors directly, so using
a Gaussian policy instead changes both exploration and centralized-critic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

try:
    from gymnasium import spaces as gym_spaces
except ImportError:  # pragma: no cover - compatibility with the legacy stack
    from gym import spaces as gym_spaces


@dataclass(frozen=True)
class ActionSpec:
    """Policy-side representation for one agent's action."""

    mode: str
    branch_sizes: tuple[int, ...] = ()

    def to_dict(self):
        return {"mode": self.mode, "branch_sizes": list(self.branch_sizes)}


class Pd:
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class PdType:
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        return torch.float32


def _gumbel_softmax(logits: torch.Tensor, uniform: torch.Tensor | None = None):
    """Match TF1's softmax(logits + Gumbel(0, 1)) sample."""

    if uniform is None:
        uniform = torch.rand_like(logits)
    finfo = torch.finfo(uniform.dtype)
    uniform = uniform.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
    gumbel = -torch.log(-torch.log(uniform))
    return torch.softmax(logits + gumbel, dim=-1)


class SoftCategoricalPdType(PdType):
    def __init__(self, ncat: int):
        self.ncat = int(ncat)

    def pdclass(self):
        return SoftCategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return [self.ncat]


class SoftCategoricalPd(Pd):
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def mode(self):
        return torch.softmax(self.logits, dim=-1)

    def sample(self, uniform: torch.Tensor | None = None):
        return _gumbel_softmax(self.logits, uniform=uniform)

    def entropy(self):
        probs = torch.softmax(self.logits, dim=-1)
        log_probs = torch.log_softmax(self.logits, dim=-1)
        return -torch.sum(probs * log_probs, dim=-1)


class SoftMultiCategoricalPdType(PdType):
    """A concatenation of independent soft-categorical branches."""

    def __init__(self, branch_sizes: Sequence[int]):
        self.branch_sizes = tuple(int(size) for size in branch_sizes)
        if not self.branch_sizes or any(size <= 0 for size in self.branch_sizes):
            raise ValueError("branch_sizes must contain positive integers")

    def pdclass(self):
        return SoftMultiCategoricalPd

    def pdfromflat(self, flat):
        return SoftMultiCategoricalPd(self.branch_sizes, flat)

    def param_shape(self):
        return [sum(self.branch_sizes)]

    def sample_shape(self):
        return [sum(self.branch_sizes)]


class SoftMultiCategoricalPd(Pd):
    def __init__(self, branch_sizes: Sequence[int], flat: torch.Tensor):
        self.branch_sizes = tuple(int(size) for size in branch_sizes)
        self.flat = flat
        self.categoricals = [
            SoftCategoricalPd(part)
            for part in torch.split(flat, self.branch_sizes, dim=-1)
        ]

    def flatparam(self):
        return self.flat

    def mode(self):
        return torch.cat([pd.mode() for pd in self.categoricals], dim=-1)

    def sample(self, uniforms: Iterable[torch.Tensor] | None = None):
        if uniforms is None:
            samples = [pd.sample() for pd in self.categoricals]
        else:
            uniforms = list(uniforms)
            if len(uniforms) != len(self.categoricals):
                raise ValueError("one uniform tensor is required per action branch")
            samples = [
                pd.sample(uniform=uniform)
                for pd, uniform in zip(self.categoricals, uniforms)
            ]
        return torch.cat(samples, dim=-1)

    def entropy(self):
        entropies = [pd.entropy() for pd in self.categoricals]
        return torch.stack(entropies, dim=-1).sum(-1)


class DiagGaussianPdType(PdType):
    def __init__(self, size: int):
        self.size = int(size)

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2 * self.size]

    def sample_shape(self):
        return [self.size]


class DiagGaussianPd(Pd):
    def __init__(self, flat: torch.Tensor):
        self.flat = flat
        self.mean, self.logstd = flat.chunk(2, dim=-1)
        self.std = torch.exp(self.logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def entropy(self):
        return torch.sum(
            self.logstd + 0.5 * (np.log(2.0 * np.pi) + 1.0), dim=-1
        )


def make_pdtype(
    ac_space,
    policy_mode: str = "official",
    branch_sizes: Sequence[int] | None = None,
):
    """Build policy distributions independently from the env API surface."""

    if policy_mode not in {"official", "gaussian"}:
        raise ValueError("unknown policy mode: {}".format(policy_mode))

    if policy_mode == "gaussian":
        if not isinstance(ac_space, gym_spaces.Box) or len(ac_space.shape) != 1:
            raise ValueError("gaussian policy mode requires a one-dimensional Box")
        return DiagGaussianPdType(ac_space.shape[0])

    if branch_sizes is not None:
        sizes = tuple(int(size) for size in branch_sizes)
        if len(sizes) == 1:
            return SoftCategoricalPdType(sizes[0])
        return SoftMultiCategoricalPdType(sizes)

    if isinstance(ac_space, gym_spaces.Discrete):
        return SoftCategoricalPdType(ac_space.n)
    if isinstance(ac_space, gym_spaces.MultiDiscrete):
        sizes = tuple(int(size) for size in np.asarray(ac_space.nvec).reshape(-1))
        return SoftMultiCategoricalPdType(sizes)
    if isinstance(ac_space, gym_spaces.Box):
        raise ValueError(
            "official policy mode needs MPE categorical branch sizes for Box actions"
        )
    raise NotImplementedError("action space {} not supported".format(type(ac_space)))
