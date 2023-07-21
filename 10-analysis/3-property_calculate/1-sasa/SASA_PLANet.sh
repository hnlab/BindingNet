#!/bin/bash
#$ -S /bin/bash
#$ -N sasa_planet
#$ -q cuda
#$ -o /pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA/SASA_PLANet.log
#$ -j y
#$ -r y
#$ -notify
#$ -l gpu=1
#$ -wd /pubhome/xli02/project/PLIM/analysis/20220812_paper/SASA

conda activate fast
printf "%s Start on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
python SASA_PLANet.py
printf "%s End on %s.\n" "[$(date +"%Y-%m-%d %H:%M:%S")]" $(hostname)
