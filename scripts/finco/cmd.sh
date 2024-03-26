#!/bin/bash
set -x # show command
set -e # Error on exception

DIR="$(
	cd "$(dirname "$(readlink -f "$0")")" || exit
	pwd -P
)"
# --load the cridentials
if [ -e $DIR/cridential.sh ]; then
	source $DIR/cridential.sh
fi

# run the command
python -m qlib.finco.cli "build an A-share stock market daily portfolio in quantitative investment and minimize the maximum drawdown."
