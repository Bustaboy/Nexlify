#!/bin/bash

# Continuous monitoring and auto-restart script
LOG_FILE="logs/monitor.log"

echo "[$(date)] Monitor started" >> $LOG_FILE

while true; do
  if ps aux | grep "train_rl_agent.py" | grep -v grep > /dev/null; then
    echo "[$(date +%H:%M:%S)] Training is running..." | tee -a $LOG_FILE
    tail -3 logs/rl_training.log | grep "Episode" | tail -1 | tee -a $LOG_FILE
    sleep 60
  else
    echo "[$(date +%H:%M:%S)] Training stopped! Restarting..." | tee -a $LOG_FILE
    cd /home/user/Nexlify
    python -u train_rl_agent.py >> logs/training_live.log 2>&1 &
    sleep 10
  fi
done
