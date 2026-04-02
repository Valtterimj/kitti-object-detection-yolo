#!/bin/bash
#SBATCH --job-name=YOLO11s-test
#SBATCH --account=project_2018697
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

PROJECT_DIR=/scratch/project_2018697/kitti-object-detection-yolo

cd "$PROJECT_DIR"

module purge
module load pytorch

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

mkdir -p logs

srun python src/kitti_object_detection_yolo/main.py