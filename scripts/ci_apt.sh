#!/usr/bin/env bash
# Installs Linux build dependencies on a CI runner.
#
# The hosted images carry a Microsoft package source that intermittently answers
# 403, which fails apt-get update and takes the whole job with it. Nothing here
# is installed from it, so it is removed before the update rather than worked
# around afterwards.
set -euo pipefail

sudo rm -f /etc/apt/sources.list.d/microsoft*.list \
           /etc/apt/sources.list.d/microsoft*.sources
sudo apt-get update -qq
sudo apt-get install -y -qq "$@"
