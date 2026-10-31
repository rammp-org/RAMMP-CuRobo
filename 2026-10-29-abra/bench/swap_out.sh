#!/bin/bash
# Undo swap_in.sh: stop the go-to-many driver and planner, start the stack's again.
docker stop goto-arm; docker start sheppy-arm
docker stop goto-planner; docker start sheppy-planner
sleep 5; docker ps --format "{{.Names}} {{.Status}}" | grep -E "sheppy-arm|sheppy-planner|goto-"
