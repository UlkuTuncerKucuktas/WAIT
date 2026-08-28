#!/bin/bash
#SBATCH -p debug
#SBATCH -C barbun
#SBATCH -A ukucuktas
#SBATCH -N 3
#SBATCH --time=1:00:00
#SBATCH --output=out/logs/%x-%j.out
#
# usage: sbatch -J s1n16 --ntasks-per-node=8 scripts/submit_scenario.sh \
#            wait.experiments.s1_barrier 16
#
# One writer node and two reader nodes.  With a single reader only one rank per
# round pays the cold fetch and the rest hit page cache, so the maximum over
# ranks -- what the barrier actually costs -- is buried in the spread of cached
# reads.  Two reader nodes put a cold fetch in every round on both.
#
# Ranks are tasks on the reading node, not processes sharing one core allocation.
# Every rank runs measure so it can take part in the barrier; rank 0 records.
set -eu
MODULE=$1
TASKS=$2
BASE=/arf/scratch/$USER/wait/$SLURM_JOB_ID
LEDGER=out/ledgers/${SLURM_JOB_NAME}.jsonl
mkdir -p "$BASE" out/ledgers out/logs

# The authoritative rank count for both phases, since srun rewrites SLURM_NTASKS
# per step.
export WAIT_SCALE=$TASKS

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export WAIT_COORDINATOR=${NODES[1]}
echo "writer=${NODES[0]} readers=${NODES[1]},${NODES[2]} x $TASKS module=$MODULE"

srun -N1 -n1 -w "${NODES[0]}" \
    python3 -m wait.run "$MODULE" prepare --base "$BASE" --ledger "$LEDGER"

# Two reader nodes put a cold fetch in every round on both, which needs at least
# one rank each.  A single-rank scenario has no second rank to give the second
# node, and asking for two nodes and one process is refused outright.
if [ "$TASKS" -lt 2 ]; then
    srun -N1 -n1 -w "${NODES[1]}" \
        python3 -m wait.run "$MODULE" measure --base "$BASE" --ledger "$LEDGER"
else
    srun -N2 -n"$TASKS" -w "${NODES[1]},${NODES[2]}" \
        python3 -m wait.run "$MODULE" measure --base "$BASE" --ledger "$LEDGER"
fi

python3 -m wait.run --reap --base "$BASE" --ledger "$LEDGER"
rm -rf "$BASE"
