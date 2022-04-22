

DOS1="altec[2,3]"
DOS2="altec[4,5]"
DOS3="altec[6,7]"
DOS4="altec[8,9]"
CUATRO1="altec[3,4,5,6]"
CUATRO1="altec[7,8,9,10]"
OCHO="altec[3,4,5,6,7,8,9,10]"

FLAGS="-q compute --exclusive --time=10:0:0"

sbatch -N 2 $FLAGS  scripts/run_512.sh 2 16
sbatch -N 2 $FLAGS  scripts/run_512.sh 2 32
sbatch -N 4 $FLAGS  scripts/run_512.sh 4 16
sbatch -N 4 $FLAGS  scripts/run_512.sh 4 32
sbatch -N 4 $FLAGS  scripts/run_512.sh 4 64
sbatch -N 8 $FLAGS  scripts/run_512.sh 8 16
sbatch -N 8 $FLAGS  scripts/run_512.sh 8 32
sbatch -N 8 $FLAGS  scripts/run_512.sh 8 64
sbatch -N 8 $FLAGS  scripts/run_512.sh 8 128

