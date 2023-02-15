#!/bin/sh
# Clear script
:> jwst.sh
# Create script
./make-sbatch-script.sh "$1" >> jwst.sh
# Give permissions
chmod 777 jwst.sh
# Submit job
sbatch jwst.sh
