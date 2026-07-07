#!/usr/bin/env bash
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PYTHONNOUSERSITE
exec /home/atos/anime-reels/f5-env/bin/python /home/atos/anime-reels/f5_generate.py "$@"
