#!/bin/bash

TESTDIR=/home/pere/results
cd $TESTDIR
set -x


# if [ -f "main" ]; then
#     rm -rf main-previous
#     mv main main-previous
# fi

if [ -f "main-aggressive" ]; then
    rm -rf main-aggressive-previous
    mv main-aggressive main-aggressive-previous
fi

cd /home/pere/ceph-main
git fetch origin
git reset --hard origin/main
rm -rf build
source scl_source enable gcc-toolset-11
./install-deps.sh
./do_cmake.sh -DCMAKE_BUILD_TYPE=RelWithDebInfo
git submodule update --init --recursive
cd build
ninja vstart -j40

# COMPRESS_MODE=aggressive TESTDIR=$TESTDIR NRUNS=16 TEST=main-aggressive python3.11 ~/tester/benchmarker.py --period 5 --outdata $TESTDIR/main-aggressive/main.json --bench="/home/pere/tester/bench-cron-script.sh" --vstart="/home/pere/tester/bench-cron-vstart.sh" --samples 1 --iterations 1 --testname "main" --warmup-iterations 0 -n
COMPRESS_MODE=none TESTDIR=$TESTDIR NRUNS=20 TEST=main-none python3.11 ~/tester/benchmarker.py --period 5 --outdata $TESTDIR/main/main.json --bench="/home/pere/tester/bench-cron-script.sh" --vstart="/home/pere/tester/bench-cron-vstart.sh" --samples 1 --iterations 1 --testname "main" --warmup-iterations 0 -n
