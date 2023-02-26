#!/bin/bash
#SBATCH --array=1-9285
#SBATCH --mem=200G
#SBATCH --cpus-per-task=18
module load TensorFlow/2.6.2-foss-2021a
source ~/virtualenvs/Dashain/bin/activate

BASEDIR=/home/t326h379/Cell_Mem_ER_Extrac_Protein

export FILENAME=$(ls ${BASEDIR}/*.fasta | sed -n ${SLURM_ARRAY_TASK_ID}p)
python ~/analyze_Cell_Mem_ER_Extrac_Protein.py
