import numpy as np
from collections import deque
from dataclasses import dataclass

# 고정 길이 deque을 만들고, 초기값으로 fill -- seed 별 최근 return/success 기록을 초기화하기 위해 사용
def create_filled_deque(maxlen, fill_value):
    return deque([fill_value] * maxlen, maxlen=maxlen)

# seed 별 성과 히스토리 저장
@dataclass
class SeedMetadata:
    seed: int
    returns: deque
    successes: deque

# 최근 Return 기록을 보고, seed 점수를 할당
def success_once_score(seed_metadata: SeedMetadata):
    returns = seed_metadata.returns
    size = 5
    # Get last 5 returns
    arr = list(returns)[-size:]
    returns_arr = np.array(arr)
    success_once_avg = (returns_arr > 0).sum() / size
    if success_once_avg == 0:
        return 2
    if success_once_avg > 0 and success_once_avg < 0.75:    # 중간 난이도 시드를 가장 높은 우선순위로
        return 3
    return 1


score_fns = {
    "success_once_score": success_once_score,
}


class ForwardCurriculumManager:
    """
    Forward Curriculum Manager (replaces SeedBasedForwardCurriculumWrapper)
    Uses pure PyTorch & NumPy tracking instead of JAX arrays and dependencies.
    """
    def __init__(
        self,
        seeds=None,
        num_seeds=1000,
        score_transform="rankmin",
        score_temperature=1e-1,
        staleness_transform="rankmin",
        staleness_temperature=1e-1,
        staleness_coef=1e-1,    # score와 staleness를 얼마나 mix할지
        score_fn: str = "success_once_score",
        rho=0,
        nu=0.5,
        num_envs=1,
    ):
        self.num_envs = num_envs
        if seeds is None:
            seeds = np.arange(0, num_seeds)
            
        self.seeds = np.array(seeds)
        self.eps_seed_to_idx = dict()
        self.seeds_db = dict()
        
        for idx, seed in enumerate(seeds):
            self.eps_seed_to_idx[seed] = idx
            self.seeds_db[seed] = SeedMetadata(
                seed=seed, 
                returns=create_filled_deque(20, 0), 
                successes=create_filled_deque(20, 0)
            )

        self.np_random = np.random.RandomState()
        
        self.seed_scores = np.zeros(len(seeds))
        self.unseen_seed_weights = np.ones(len(seeds))
        
        self.score_transform = score_transform
        self.score_temperature = score_temperature

        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.staleness_coef = staleness_coef
        self.seed_staleness = np.zeros(len(seeds))

        self.rho = rho
        self.nu = nu
        self.score_fn = score_fns[score_fn]

        # Init unseen weights mapping
        all_seed_indices = np.arange(len(self.seeds))
        for i in all_seed_indices[:: self.num_envs]:
            seed_indices = all_seed_indices[i : i + self.num_envs]
            if len(seed_indices) < self.num_envs:
                padding = np.array([0] * (self.num_envs - len(seed_indices)))
                seed_indices = np.concatenate([seed_indices, padding])
            for reset_seed_idx in seed_indices:
                self.unseen_seed_weights[reset_seed_idx] = 0
                self.seed_scores[reset_seed_idx] = 2

    def record_episode(self, eps_seed, final_return, success):
        eps_seed_idx = self.eps_seed_to_idx[eps_seed]
        self.unseen_seed_weights[eps_seed_idx] = 0  # 한 번이라도 사용된 seed는 seen 처리
        seed_metadata = self.seeds_db[eps_seed]
        seed_metadata.returns.append(final_return)
        seed_metadata.successes.append(success)
        self.seed_scores[eps_seed_idx] = self.score_fn(seed_metadata)

    def sample_seeds(self, count):
        num_unseen_seeds = self.unseen_seed_weights.sum()
        num_seen_seeds = len(self.seeds) - num_unseen_seeds
        proportion_seen = num_seen_seeds / len(self.seeds)
        
        if num_unseen_seeds == 0:
            seed_indices = self._sample_seen_seeds(count)
        else:
            ps = self.np_random.rand(count)
            unseen_seed_indices = self._sample_unseen_seeds(count)
            if proportion_seen <= self.rho:
                seed_indices = unseen_seed_indices
            else:
                seen_seed_indices = self._sample_seen_seeds(count)
                seen_selected = seen_seed_indices[ps >= self.nu]
                unseen_selected = unseen_seed_indices[ps < self.nu]
                seed_indices = np.concatenate([unseen_selected, seen_selected])
                # In case of size mismatch logic
                if len(seed_indices) > count:
                    seed_indices = seed_indices[:count]
                elif len(seed_indices) < count:
                    padding = self._sample_seen_seeds(count - len(seed_indices))
                    seed_indices = np.concatenate([seed_indices, padding])

        self._update_staleness(seed_indices)
        return self.seeds[seed_indices], seed_indices

    def _update_staleness(self, seed_indices):
        for seed in seed_indices:
            self.seed_staleness += 1
            self.seed_staleness[seed] = 0

    def _score_transform_fn(self, transform, temperature, scores):
        if transform == "constant":
            weights = np.ones_like(scores)
        elif transform == "identity":
            return scores
        elif transform == "rank":
            temp = np.flip(np.argsort(scores))
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / (ranks ** (1.0 / temperature))
        elif transform == "rankmin":
            def rankmin(x):
                u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
                csum = np.zeros_like(counts)
                csum[1:] = counts[:-1].cumsum()
                return csum[inv]
            ranks = len(scores) - rankmin(scores) + 1
            weights = (1 / ranks) ** (1 / temperature)
        elif transform == "power":
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1.0 / temperature)
        elif transform == "softmax":
            weights = np.exp(np.array(scores) / temperature)
        return weights

    def _sample_seen_seeds(self, count):
        weights = self._score_transform_fn(self.score_transform, self.score_temperature, self.seed_scores)
        weights = weights * (1 - self.unseen_seed_weights)
        z = weights.sum()
        if z > 0:
            weights = weights / z
            
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform_fn(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1 - self.unseen_seed_weights)
            z = staleness_weights.sum()
            if z > 0:
                staleness_weights = staleness_weights / z
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights

        z = weights.sum()
        if z > 0:
            weights = weights / z
        else:
            weights = np.ones_like(weights) / len(weights)
            
        seed_indices = self.np_random.choice(range(len(self.seeds)), size=count, p=weights)
        return seed_indices

    def _sample_unseen_seeds(self, count):
        num_unseen_seeds = self.unseen_seed_weights.sum()
        probs = self.unseen_seed_weights / num_unseen_seeds
        seed_indices = self.np_random.choice(range(len(self.seeds)), size=count, p=probs)
        return seed_indices
