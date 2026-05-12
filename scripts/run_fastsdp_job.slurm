#!/bin/bash
#SBATCH --job-name=fast_sdp
#SBATCH -A llc@cpu                      # Compte CPU (MOSEK est CPU-only)
#SBATCH --partition=cpu_p1              # Partition CPU
#SBATCH --cpus-per-task=4              # CPUs dispo (MOSEK utilise 4 threads)
#SBATCH --time=2:00:00                 # Temps max par job (1h30 solver + 30min marge)
#SBATCH --chdir=/lustre/fswork/projects/rech/llc/uvq13au/FastSDPCertification
#SBATCH --output=slurm-%x-%j.out       # Log SLURM standard (à la racine du projet)
#SBATCH --error=slurm-%x-%j.err        # Log SLURM erreur

# Charge les modules nécessaires
module purge
source ~/.bashrc
module load pytorch-gpu/py3/2.0.0
module unload openmpi
module load openmpi/4.1.1

# Active ton venv
source /lustre/fswork/projects/rech/llc/uvq13au/venvs/fastsdp/bin/activate

# Crée le répertoire de logs si besoin
DATE=$(date +%Y_%m_%d_%Hh%M_%Ss)
LOG_DIR="results/benchmark/${NETWORK_NAME}-${EPSILON}/${DATE}_${NAME_RUN}"
mkdir -p "$LOG_DIR"

# Redirige stdout et stderr
exec > >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.err" >&2)

echo "Network: $NETWORK_NAME"
echo "Run name: $NAME_RUN"
echo "Job ID: $SLURM_JOB_ID"

# Appel du script métier avec les arguments du SLURM
if [ -n "$START" ] && [ -n "$END" ]; then
    python src/certification_problem.py "$NETWORK_NAME" "$NAME_RUN" --start "$START" --end "$END"
else
    python src/certification_problem.py "$NETWORK_NAME" "$NAME_RUN"
fi