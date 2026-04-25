Recommendations

- Main cause is `time_limit` (109/126 short runs).
- Keep `target_epoch=199` as the acceptance rule; every dropped run here truly ended before epoch 199.
- For CIFAR10 jobs, the 12-hour walltime is too aggressive when one Slurm task runs 5 seeds sequentially.
- Recommended: either raise walltime to at least 24 hours, or set `num_experiments=1` and fan out seeds through the array dimension.
- For CIFAR100, the slowest incomplete groups are: 60 clients + FangAttack (48), 40 clients + FangAttack (25), 60 clients + MinSum (6).
- Recommended: split 40/60-client FangAttack jobs into one seed per task, or increase walltime beyond 24 hours for those subsets.
- `15` runs ended with `Bus error`; rerunning them without addressing the runtime fault is unlikely to help.
- Python exceptions need code-side fixes before rerun. Seen examples: IndexError: slice() cannot be applied to a 0-dim tensor., OSError: [Errno 5] Input/output error.

Practical Slurm changes

- Change `num_experiments=5` to `num_experiments=1` in the submit scripts.
- Add seeds to the array grid instead of looping over them inside a single job.
- Keep 12h only for clearly fast CIFAR10 subsets; use a longer walltime for FangAttack-heavy jobs.
- For CIFAR100 with 60 clients, treat FangAttack as a separate long-job class.
