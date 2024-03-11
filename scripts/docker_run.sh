#!/bin/bash
set -euxo pipefail

docker run --pull=always --rm -it -v ~/data/qlinks:/home/scripts/data tanlin2013/qlinks
