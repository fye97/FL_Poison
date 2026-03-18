from flpoison.aggregators import aggregator_registry
from flpoison.aggregators.aggregatorbase import AggregatorBase
from flpoison.aggregators.aggregator_utils import L2_distances, krum_compute_scores


@aggregator_registry
class Krum(AggregatorBase):
    """
    [Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent](https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html) - NeurIPS '17
    Krum first compute the score of each update, which is the sum of the n-f-1 smallest Euclidean distances to the other updates. Then it selects the update with the smallest score as the aggregated update.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        enable_check (bool): whether to enable the check of the number of Byzantine clients
        """
        self.default_defense_params = {"enable_check": False}
        self.update_and_set_attr()

    def aggregate(self, updates, **kwargs):
        num_clients = len(updates)
        if self.enable_check:
            if 2 * self.args.num_adv + 2 >= num_clients:
                raise ValueError(
                    f"num_byzantine should be meet 2f+2 < n, got 2*{self.args.num_adv}+2 >= {num_clients}."
                )
        with self.profile_substage("defense"):
            distances = L2_distances(updates)
            scores = [(i, krum_compute_scores(distances, i, num_clients, self.args.num_adv))
                      for i in range(num_clients)]
            sorted_scores = sorted(scores, key=lambda x: x[1])
            selected_idx = sorted_scores[0][0]
        with self.profile_substage("aggregate"):
            return updates[selected_idx]


def krum(updates, num_byzantine=0, return_index=False, enable_check=False):
    num_clients = len(updates)
    if enable_check:
        if 2 * num_byzantine + 2 >= num_clients:
            raise ValueError(
                f"num_byzantine should be meet 2f+2 < n, got 2*{num_byzantine}+2 >= {num_clients}."
            )
    # calculate euclidean distance between clients
    distances = L2_distances(updates)
    # calculate client i's score
    scores = [(i, krum_compute_scores(distances, i, num_clients, num_byzantine))
              for i in range(num_clients)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    if return_index:
        return sorted_scores[0][0]
    else:
        return updates[sorted_scores[0][0]]
