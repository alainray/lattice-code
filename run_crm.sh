#!/bin/bash
#SBATCH --job-name=run_comp
#SBATCH --output=exp_logs/%x_%j.out
#SBATCH --error=exp_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --account=ACCOUNT
#SBATCH --partition=gpu-partition
#SBATCH --nodelist=cluster-a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --comment="Experimento: CRM+ED"
#SBATCH --qos=regular
# Activar entorno si es necesario
# El /tmp del nodo se llena con archivos de otros usuarios y los workers
# del DataLoader mueren con OSError 28; los temporales van al scratch.
export TMPDIR=/scratch/USER/tmp
mkdir -p "$TMPDIR"
source ~/venv/bin/activate

cd ~/scalable-compositional-generalization

./crm_runner.sh $ds crm ${model:-all} $seeds

wait

echo "Finished with job $SLURM_JOBID (seed=$SEED)"
