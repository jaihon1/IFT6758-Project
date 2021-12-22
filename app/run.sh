#!/bin/bash

docker run -it -p 127.0.0.1:5000:5000/tcp --env COMET_API_KEY=$COMET_API_KEY -d ift6758/serving:1.0.0
docker run -it -p 127.0.0.1:4000:4000/tcp --env COMET_API_KEY=$COMET_API_KEY -d ift6758/jupyter:1.0.0
