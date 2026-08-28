#!/bin/bash
#SBATCH -p debug
#SBATCH -C barbun
#SBATCH -A ukucuktas
#SBATCH -N 2
#SBATCH --time=0:20:00
#SBATCH --output=out/logs/%x-%j.out
set -eu
BASE=/arf/scratch/$USER/inline/$SLURM_JOB_ID
mkdir -p "$BASE" out/logs
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "writer=${NODES[0]} reader=${NODES[1]}"
srun -N1 -n1 -w "${NODES[0]}" python3 scripts/probe_inline.py --phase write --base "$BASE"
srun -N1 -n1 -w "${NODES[1]}" python3 scripts/probe_inline.py --phase read  --base "$BASE"
rm -rf "$BASE"
