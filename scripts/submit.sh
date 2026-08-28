#!/bin/bash
#SBATCH -p debug
#SBATCH -C barbun
#SBATCH -A ukucuktas
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=out/logs/%x-%j.out
#
# usage: sbatch -J a2 scripts/submit.sh wait.experiments.a2_tier_benefit
#
# A reader on the writer's own node sees the client cache: DoM measured 78.1 us
# against OST at 77.8, no difference at all.  The two srun steps are the point.
set -eu
MODULE=$1
BASE=/arf/scratch/$USER/wait/$SLURM_JOB_ID
LEDGER=out/ledgers/${SLURM_JOB_NAME}.jsonl
mkdir -p "$BASE" out/ledgers out/logs

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "writer=${NODES[0]} reader=${NODES[1]} module=$MODULE"

srun -N1 -n1 -w "${NODES[0]}" python3 -m wait.run "$MODULE" prepare --base "$BASE" --ledger "$LEDGER"
srun -N1 -n1 -w "${NODES[1]}" python3 -m wait.run "$MODULE" measure --base "$BASE" --ledger "$LEDGER"

python3 -m wait.run --reap --base "$BASE" --ledger "$LEDGER"
rm -rf "$BASE"
