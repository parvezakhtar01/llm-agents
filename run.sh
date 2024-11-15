#!/bin/bash

# Export the PYTHONPATH to include the project root
export PYTHONPATH=$PWD:$PYTHONPATH

# Run the server
cd app
uvicorn main:app --reload --port 8000