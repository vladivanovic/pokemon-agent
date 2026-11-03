from __future__ import annotations

import json
import logging
import os

import numpy as np
from einops import rearrange, reduce
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

_NUMERIC = (int, float, np.integer, np.floating)


def merge_dicts(dicts):
    """Mean + raw distribution for every scalar key present in any dict."""
    sums, counts, distrib = {}, {}, {}
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            # bool is a subclass of int; count it, but never a numpy array
            if isinstance(v, _NUMERIC) and not isinstance(v, bool):
                v = float(v)
            elif isinstance(v, bool):
                v = float(v)
            else:
                continue
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1
            distrib.setdefault(k, []).append(v)
    means = {k: sums[k] / counts[k] for k in sums}
    return means, {k: np.asarray(v, dtype=np.float32) for k, v in distrib.items()}


def _downsample_max(img, factor):
    if factor <= 1:
        return img
    h, w = img.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    if h2 == 0 or w2 == 0:
        return img
    return reduce(img[:h2, :w2], "(h p) (w q) -> h w", "max", p=factor, q=factor)


def _tile(maps, cols=None):
    """Grid-tile a stack of maps, padding to fill the last row.

    The old 'r=2' rearrange raised whenever n_envs was odd or 1.
    """
    n, h, w = maps.shape
    cols = cols or max(1, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    if rows * cols > n:
        pad = np.zeros((rows * cols - n, h, w), dtype=maps.dtype)
        maps = np.concatenate([maps, pad], axis=0)
    return rearrange(maps, "(r c) h w -> (r h) (c w)", r=rows, c=cols)


class TensorboardCallback(BaseCallback):
    """Periodic env telemetry.

    Sampling is interval-based rather than episode-end-based: VecEnv
    auto-resets on done, which wipes agent_stats and explore_map before a
    terminal-boundary callback could read them.
    """

    def __init__(self, log_dir, verbose=0, log_freq=2048, image_freq=None,
                 map_downsample=2, max_map_tiles=16):
        super().__init__(verbose)
        self.log_dir = str(log_dir)
        self.log_freq = int(log_freq)
        self.image_freq = int(image_freq or self.log_freq * 8)
        self.map_downsample = int(map_downsample)
        self.max_map_tiles = int(max_map_tiles)
        self.writer = None

    def _on_training_start(self) -> None:
        if self.writer is not None:
            return
        # self.logger.dir is the active TB run dir; writing there keeps
        # histograms in the same run instead of a sibling 'histogram' run
        target = getattr(self.logger, "dir", None) or os.path.join(
            self.log_dir, "histogram")
        try:
            self.writer = SummaryWriter(log_dir=target)
        except Exception:
            logger.warning("could not open SummaryWriter at %s", target,
                           exc_info=True)

    def _get_attr(self, name):
        """get_attr, falling back through gymnasium wrappers.

        Gymnasium >=1.0 dropped implicit attribute forwarding on Wrapper, so
        get_attr can miss env fields when StreamWrapper is applied.
        """
        try:
            return self.training_env.get_attr(name)
        except (AttributeError, Exception):
            try:
                return self.training_env.env_method("get_wrapper_attr", name)
            except Exception:
                logger.debug("attr %s unavailable", name, exc_info=True)
                return None

    def _on_step(self) -> bool:
        try:
            if self.n_calls % self.log_freq == 0:
                self._log_scalars()
            if self.n_calls % self.image_freq == 0:
                self._log_images()
                self._log_flags()
        except Exception:
            # telemetry must never take down a training run
            logger.warning("tensorboard callback step failed", exc_info=True)
        return True

    def _log_scalars(self):
        all_stats = self._get_attr("agent_stats")
        if all_stats:
            latest = [s[-1] for s in all_stats if s]  # deques may be empty
            if latest:
                means, distributions = merge_dicts(latest)
                for key, val in means.items():
                    self.logger.record(f"env_stats/{key}", val)
                for key, distrib in distributions.items():
                    self.logger.record(f"env_stats_max/{key}", float(distrib.max()))
                    if self.writer is not None:
                        self.writer.add_histogram(
                            f"env_stats_distribs/{key}", distrib,
                            self.num_timesteps)  # timesteps, not n_calls

        totals = self._get_attr("total_reward")
        if totals:
            arr = np.asarray([float(t) for t in totals], dtype=np.float32)
            self.logger.record("env_stats/total_reward_mean", float(arr.mean()))
            self.logger.record("env_stats/total_reward_max", float(arr.max()))

        wins = self._get_attr("battle_won_count")
        if wins:
            self.logger.record("env_stats/battle_wins_mean",
                               float(np.mean(wins)))

    def _log_images(self):
        maps = self._get_attr("explore_map")
        if not maps:
            return
        # 64 full-res global maps is ~12 MB through the subproc pipes per call
        maps = np.asarray(maps[:self.max_map_tiles], dtype=np.uint8)
        if maps.ndim != 3:
            return

        union = reduce(maps, "f h w -> h w", "max")
        self.logger.record(
            "trajectory/explore_sum",
            Image(np.ascontiguousarray(union), "HW"),
            exclude=("stdout", "log", "json", "csv"))

        small = np.stack([_downsample_max(m, self.map_downsample) for m in maps])
        self.logger.record(
            "trajectory/explore_map",
            Image(np.ascontiguousarray(_tile(small)), "HW"),
            exclude=("stdout", "log", "json", "csv"))

    def _log_flags(self):
        flag_dicts = self._get_attr("current_event_flags_set")
        if not flag_dicts:
            return
        merged = {}
        for d in flag_dicts:
            if isinstance(d, dict):
                merged.update(d)
        if merged:
            self.logger.record("trajectory/all_flags", json.dumps(merged))

    def _on_training_end(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
                self.writer.close()
            finally:
                self.writer = None