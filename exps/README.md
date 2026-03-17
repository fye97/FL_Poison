# Experiments

`exps/` now has one job:

- `specs/*.yaml`: define experiment matrices
- `launch.py`: `list`, `plan`, `local`, `cc`, `worker`
- `slurm_array_entry.sh`: generic Compute Canada array entrypoint

Examples:

```bash
python exps/launch.py list
python exps/launch.py plan cifar10
python exps/launch.py local cifar10 --ids 0-7 --jobs 1
python exps/launch.py cc cifar10 --chunk-size 32
```

The design intent is strict separation:

- Experiment content lives in `specs/*.yaml`
- Execution framework lives in `launch.py`
- Backend choice is only `local` vs `cc`
