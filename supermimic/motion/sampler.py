"""Return-based reset sampling (SuperMimic, Appendix D.1).

The motion timeline is cut into 1 s bins. Each bin keeps a running estimate
G_b of the discounted return earned from it. Resets mix

* 50 % uniform over bins,
* 40 % from softmax_b(-25 G_b / G_max),
* 10 % in proportion to each bin's accumulated past softmax weight,

so hard bins are practiced and a hard motion keeps some practice after it is
learned. Returns are tracked per env over a window of ``window`` steps from the
bin it started in; a window that ends in a timeout (clip end) is normalised by
the best return attainable over its actual length, so short clip tails are not
mistaken for failures, while a termination is normalised by the full window.
"""

from __future__ import annotations

import torch

from supermimic.motion.library import MotionLibrary


class ReturnBasedSampler:
    def __init__(
        self,
        library: MotionLibrary,
        num_envs: int,
        max_step_reward: float,
        gamma: float = 0.99,
        window: int = 250,
        mix: tuple[float, float, float] = (0.5, 0.4, 0.1),
        temperature: float = 25.0,
        ema: float = 0.1,
    ):
        self.lib = library
        self.device = library.device
        self.gamma = gamma
        self.window = window
        self._mix = torch.tensor(mix, device=self.device)
        self.temperature = temperature
        self.ema = ema
        self.max_step_reward = max_step_reward

        nb = library.num_bins
        self.bin_score = torch.zeros(nb, device=self.device)  # G_b / G_max in [0, 1]; 0 = pessimistic start
        self.bin_visits = torch.zeros(nb, device=self.device)
        self.past_weight = torch.full((nb,), 1.0 / nb, device=self.device)

        self.start_bin = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.ret = torch.zeros(num_envs, device=self.device)
        self.discount = torch.ones(num_envs, device=self.device)
        self.steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    def _gmax(self, n: torch.Tensor) -> torch.Tensor:
        return self.max_step_reward * (1.0 - self.gamma ** n.float()) / (1.0 - self.gamma)

    def softmax_weights(self) -> torch.Tensor:
        return torch.softmax(-self.temperature * self.bin_score, dim=0)

    def sample_frames(self, n: int) -> torch.Tensor:
        """Draw ``n`` flat frame indices into the motion library."""
        nb = self.lib.num_bins
        soft = self.softmax_weights()
        self.past_weight += soft
        # draw from all three and pick per sample, avoiding host syncs on per-source counts
        source = torch.multinomial(self._mix, n, replacement=True)
        uniform = torch.randint(0, nb, (n,), device=self.device)
        hard = torch.multinomial(soft, n, replacement=True)
        past = torch.multinomial(self.past_weight, n, replacement=True)
        bins = torch.where(source == 0, uniform, torch.where(source == 1, hard, past))
        lo, hi = self.lib.bin_start[bins], self.lib.bin_end[bins]
        return lo + (torch.rand(n, device=self.device) * (hi - lo + 1).float()).long().clamp_max(hi - lo)

    def begin(self, env_ids: torch.Tensor, frames: torch.Tensor):
        self.start_bin[env_ids] = self.lib.frame_bin[frames]
        self.ret[env_ids] = 0.0
        self.discount[env_ids] = 1.0
        self.steps[env_ids] = 0

    def update(self, reward: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, frames: torch.Tensor):
        """Accumulate one step; close windows that ended. ``frames`` are the envs' current frames."""
        self.ret += self.discount * reward
        self.discount *= self.gamma
        self.steps += 1
        window_full = self.steps >= self.window
        closed = (terminated | truncated | window_full).float()
        # masked over all envs rather than indexed by the closed ones, so there is no host sync
        n_eff = torch.where(terminated, torch.full_like(self.steps, self.window), self.steps)
        score = (self.ret / self._gmax(n_eff)).clamp(0.0, 1.0)
        # EMA towards the new estimate; scatter-mean if several envs close the same bin
        total = torch.zeros_like(self.bin_score).index_add_(0, self.start_bin, torch.where(closed > 0, score, 0.0))
        count = torch.zeros_like(self.bin_score).index_add_(0, self.start_bin, closed)
        mean = total / count.clamp_min(1.0)
        new = torch.where(self.bin_visits == 0, mean, (1 - self.ema) * self.bin_score + self.ema * mean)
        self.bin_score.copy_(torch.where(count > 0, new, self.bin_score))
        self.bin_visits += count
        # windows that filled up mid-episode restart from the bin the env is in now
        cont = window_full & ~(terminated | truncated)
        self.start_bin.copy_(torch.where(cont, self.lib.frame_bin[frames], self.start_bin))
        self.ret.masked_fill_(cont, 0.0)
        self.discount.masked_fill_(cont, 1.0)
        self.steps.masked_fill_(cont, 0)

    def stats(self) -> dict[str, float]:
        visited = self.bin_visits > 0
        return {
            "sampler/bins_visited_frac": float(visited.float().mean()),
            "sampler/mean_bin_score": float(self.bin_score[visited].mean()) if visited.any() else 0.0,
            "sampler/min_bin_score": float(self.bin_score[visited].min()) if visited.any() else 0.0,
        }
