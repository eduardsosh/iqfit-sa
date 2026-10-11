# IQ Fit – Simulated Annealing

Solves the **IQ Fit** puzzle (SmartGames) with hand-written Simulated Annealing
in pure Python (standard library only). Optimisation course assignment.

```bash
python solve.py instances/std_5x10_k10_1.json --seed 1 --time 60   # solve one instance
python generate_instances.py                                         # regenerate instances
python run_experiments.py --runs 10 --time 60                        # run all experiments
```

The SA is in `iqfit/sa.py`. `iqfit/exact.py` is an exact solver that is used
only to generate test instances. Instances are in `instances/`, and results
are in [`results/summary.md`](results/summary.md).
