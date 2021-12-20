#!/bin/bash

docker build -t ift6758/serving:1.0.0 -f Dockerfile.serving .
docker build -t ift6758/jupyter:1.0.0 -f Dockerfile.jupyter .