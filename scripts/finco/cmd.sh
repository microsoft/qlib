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
python -m qlib.finco.cli "please help me build a low turnover strategy that focus more on longterm return"
