



FLAGS="-q compute --exclusive --time=10:0:0"

sbatch -N 2 $FLAGS scripts/run_256.sh -n 2 -bs 16
sbatch -N 2 $FLAGS scripts/run_256.sh -n 2 -bs 32
sbatch -N 4 $FLAGS scripts/run_256.sh -n 4 -bs 16
sbatch -N 4 $FLAGS scripts/run_256.sh -n 4 -bs 32
sbatch -N 4 $FLAGS scripts/run_256.sh -n 4 -bs 64
sbatch -N 8 $FLAGS scripts/run_256.sh -n 8 -bs 16
sbatch -N 8 $FLAGS scripts/run_256.sh -n 8 -bs 32
sbatch -N 8 $FLAGS scripts/run_256.sh -n 8 -bs 64
sbatch -N 8 $FLAGS scripts/run_256.sh -n 8 -bs 128

