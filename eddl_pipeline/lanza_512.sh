


FLAGS="-q compute --exclusive --time=10:0:0"

BS=16
for PROCS in 1 2 4 8
do
sbatch -N $PROCS $FLAGS  scripts/run_512.sh -n $PROCS -bs $BS
done

BS=32
for PROCS in 1 2 4 8
do
sbatch -N $PROCS $FLAGS  scripts/run_512.sh -n $PROCS -bs $BS
done

BS=64
for PROCS in 1 2 4 8
do
sbatch -N $PROCS $FLAGS  scripts/run_512.sh -n $PROCS -bs $BS
done

BS=128
for PROCS in 1 2 4 8
do
sbatch -N $PROCS $FLAGS  scripts/run_512.sh -n $PROCS -bs $BS
done


