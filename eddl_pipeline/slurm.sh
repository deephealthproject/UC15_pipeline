BS=$1
sbatch -N 2 -q compute --nodelist=altec[2-3] --exclusive scripts/run_test_distr.sh 2 $BS
sbatch -N 4 -q compute --nodelist=altec[2-4],altec[7] --exclusive scripts/run_test_distr.sh 4 $BS
sbatch -N 6 -q compute --nodelist=altec[2-4],altec[7-9] --exclusive scripts/run_test_distr.sh 6 $BS
