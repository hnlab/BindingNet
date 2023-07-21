#!/bin/bash

INPUT=$(ls INDEX_general_PL_data.2019)
awk '{if($1!~/#/) print $1}' ${INPUT} > $(dirname ${INPUT})/PDBIDs_$(basename ${INPUT})

