#!/bin/bash

# Specify the directory where the Slurm output and error files are located
output_dir="/store/DAMTPEGLEN/mw894/slurm/Charlesworth2015"

# Initialize a counter for failed jobs
failed_jobs=0

# Loop through the output files
for file in "$output_dir"/slurm-*.out; do
    # Extract the job ID from the file name
    job_id=$(basename "$file" | sed 's/slurm-//;s/\.out//')


    if grep -q "CANCELLED" "$file"; then
        echo "Job $job_id failed. Check $file for details."
        ((failed_jobs++))
    fi

done

# Print the number of failed jobs
echo "Total failed jobs: $failed_jobs"