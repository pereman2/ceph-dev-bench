#!/bin/bash

set -ex
TESTDIR=/home/pere/results
NRUNS=8

do_test() {
    FOLDER=$1
    COMPRESS_MODE=$2
    TEST=$3
    BRANCH=$4
    cd $FOLDER
    # clear previous runs
    rm $TESTDIR/$TEST/* -f
    git fetch origin
    git fetch pere || true;
    git fetch igor || true;
    # git reset --hard $BRANCH
    # git pull --rebase origin main
    git submodule update --init --recursive
    ninja vstart -j80
    ../src/stop.sh
    EXTRA_DEPLOY_OPTIONS=$EXTRA_DEPLOY_OPTIONS COMPRESS_MODE=$COMPRESS_MODE TESTDIR=$TESTDIR NRUNS=$NRUNS TEST=$TEST python3.11 /home/pere/tester/benchmarker.py --period 5 --outdata $TESTDIR/$TEST/main.json --bench="/home/pere/tester/bench.sh" --vstart="/home/pere/tester/do_vstart.sh" --samples 1 --iterations 1 --testname "$TEST" --warmup-iterations 0 -n
    # EXTRA_DEPLOY_OPTIONS=$EXTRA_DEPLOY_OPTIONS COMPRESS_MODE=$COMPRESS_MODE TESTDIR=$TESTDIR NRUNS=$NRUNS TEST=$TEST python3.11 /home/pere/tester/benchmarker.py --period 5 --outdata $TESTDIR/$TEST/main.json --bench="/home/pere/tester/bench.sh" --vstart="/home/pere/tester/do_vstart.sh" --samples 1 --iterations 1 --testname "$TEST" --warmup-iterations 0
}

# needs extra space?
# export EXTRA_DEPLOY_OPTIONS="-o bluestore_allocator=hybrid_btree2 "
# export EXTRA_DEPLOY_OPTIONS="-o bluestore_allocator=hybrid "

do_test /home/pere/ceph-main/build none main-none-8-test origin/main
# do_test /home/pere/ceph/build none igor-bufferspace igor/wip-ifed-onode-space-final

#do_test /home/pere/ceph/build none wal-fsync-8 pere/wal-fsync
# do_test /home/pere/ceph/build none batch-db-get2 pere/batch-db-get

# testname=igore-bufferspace
# outplots=~/results-plots
# mkdir -p $outplots/$testname
# ~/tester/format-incerta.sh ~/results/main-none-8/ ~/results/igor-bufferspace $outplots/$testname
