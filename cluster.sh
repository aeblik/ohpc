#!/bin/bash
#SBATCH --chdir=/cfs/earth/scratch/staniluk/OHPC
#SBATCH --time=01-00:00:00
#SBATCH --job-name=annealing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=earth-3
#SBATCH --mem=128G
#SBATCH --constraint=rhel8
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=staniluk@students.zhaw.ch
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Lade Python-Module
module purge
module load DefaultModules
module load python/3.9.12-pe5.34

# Virtuelle Umgebung erstellen und aktivieren
echo "Erstelle und aktiviere virtuelle Umgebung..."
python3 -m venv venv
source venv/bin/activate

# Aktualisiere pip und installiere Pakete
echo "Installiere benötigte Pakete..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas matplotlib

# Überprüfen der installierten Pakete
echo "Installierte Pakete:"
pip list

# Führe das Python-Skript aus
echo "Starte Python-Skript..."
python cluster.py
