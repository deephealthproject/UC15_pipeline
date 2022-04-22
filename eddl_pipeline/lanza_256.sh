



FLAGS="-q compute --exclusive --time=10:0:0"

sbatch -N 2 $FLAGS scripts/run_256.sh 2 16
sbatch -N 2 $FLAGS scripts/run_256.sh 2 32
sbatch -N 4 $FLAGS scripts/run_256.sh 4 16
sbatch -N 4 $FLAGS scripts/run_256.sh 4 32
sbatch -N 4 $FLAGS scripts/run_256.sh 4 64
sbatch -N 8 $FLAGS scripts/run_256.sh 8 16
sbatch -N 8 $FLAGS scripts/run_256.sh 8 32
sbatch -N 8 $FLAGS scripts/run_256.sh 8 64
sbatch -N 8 $FLAGS scripts/run_256.sh 8 128

