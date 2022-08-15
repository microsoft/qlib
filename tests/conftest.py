import os
import sys

"""Ignore RL tests on non-linux platform."""
collect_ignore = []

if sys.platform != "linux":
    for root, dirs, files in os.walk("rl"):
        for file in files:
            collect_ignore.append(os.path.join(root, file))
