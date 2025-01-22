#!/bin/bash

########################################################################################################################
# INSTALL
# Follow the installation guide https://www.jetbrains.com/help/pycharm/remote-development-a.html#gateway to install
# the JetBrains Gateway
########################################################################################################################

# Configuration
REMOTE_USER="amarti51"
REMOTE_HOST="unil"
REMOTE_SCRIPT_PATH="/users/amarti51/projects/virtues/create-remote-session.sbatch"
LOG_DIR="/work/FAC/FBM/DBC/mrapsoma/prometex/logs" # Logs directory on the remote machine

# Submit the job and capture the job ID
JOB_ID=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "sbatch ${REMOTE_SCRIPT_PATH}" | awk '{print $4}')
OUTPUT_LOG_PATTERN="remote-session-$JOB_ID.log" # Pattern to match the log file

if [[ -z "$JOB_ID" ]]; then
    echo "Failed to submit the job."
    exit 1
fi

echo "Job submitted with ID: $JOB_ID"

# Monitor the log for the desired link
echo "Waiting for the link to appear in the logs..."

while true; do
    # Fetch the latest log content and search for the link
    LINK=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "grep 'Http link: https://code-with-me.jetbrains.com' ${LOG_DIR}/${OUTPUT_LOG_PATTERN}" 2>/dev/null)

    # Check if the grep command succeeded and a link was found
    if [[ $? -eq 0 && -n "$LINK" ]]; then
        echo "Link found:"
        echo "$LINK" # Print the entire matched line
        # Extract the actual URL using awk and print it
        ACTUAL_LINK=$(echo "$LINK" | awk -F' ' '{print $NF}')
        echo "Extracted URL: $ACTUAL_LINK"
        break
    fi

    # Wait for a few seconds before checking again
    sleep 5
done