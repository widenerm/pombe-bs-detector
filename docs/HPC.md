# Running BS-Detector on a SLURM cluster

Segmentation is the only step that needs a GPU, and it splits cleanly across several:
each array task segments its own block of frames into the same `working/masks/` folder. The
scripts in [`hpc/`](../hpc/) chain the four steps with job dependencies. The defaults
(partitions `gpu` and `short`, module `python/3.13.5`) are for Northeastern's Explorer
cluster; set the variables below for another cluster.

## One-time setup

On a login node:

```bash
git clone https://github.com/widenerm/pombe-bs-detector.git
cd pombe-bs-detector
bash hpc/setup_env.sh          # CUDA PyTorch + bs-detector into /scratch/$USER/bsd-env
```

| Variable | Default | Meaning |
|---|---|---|
| `BSD_ENV` | `/scratch/$USER/bsd-env` | where the Python environment lives |
| `BSD_PYTHON_MODULE` | `python/3.13.5` | module that provides Python ≥ 3.10 (`""` to skip `module load`) |
| `BSD_TORCH_INDEX` | `https://download.pytorch.org/whl/cu126` | PyTorch wheels for your CUDA version |
| `BSD_GPU_PARTITION`, `BSD_CPU_PARTITION` | `gpu`, `short` | partitions for segmentation and for the CPU steps |
| `BSD_GPU_TIME` | `01:30:00` | time limit per segmentation task |

## Analyse a movie

Copy the movie to scratch first. Then, from the folder where you want the logs:

```bash
bash hpc/submit_movie.sh /scratch/$USER/movies/my_movie.nd2 /scratch/$USER/runs/my_movie 4
```

This submits, each waiting for the previous one:

1. `bsd prepare` on a CPU node: reads the movie and resamples it if needed;
2. `bsd segment --block K/4` as a 4-task GPU array, one block of frames per task;
3. `bsd track` on a CPU node (5–60 min; the whole movie is held in memory, so it asks for 64 GB);
4. `bsd growth`.

Extra options go to `bsd prepare`, for example `... my_movie.nd2 RUN 4 --position 2`.
A 271-frame 2304 × 2304 movie needs 1–3 GPU-hours in total, so under an hour per task on
four GPUs, plus queue time.

**If a task hits its time limit or a node fails,** run the same `submit_movie.sh` command
again. Frames already segmented are skipped, so only the missing ones are redone.
Masks from different GPUs (or a Mac) can be mixed: each mask records its device. On the
movies checked, CUDA and Apple-GPU masks were pixel-identical on most frames and differed
by 1 pixel on the rest.

## Individual steps

`hpc/bsd_job.sbatch` runs any `bsd` command in the environment, so steps can also be
submitted by hand:

```bash
sbatch -p short hpc/bsd_job.sbatch prepare movie.nd2 --out RUN
sbatch -p gpu --gres=gpu:1 --array=0-3 hpc/bsd_job.sbatch segment RUN --block slurm
sbatch -p short --mem=64G hpc/bsd_job.sbatch track RUN
```

`--block slurm` becomes the task's own block (`$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT`).
