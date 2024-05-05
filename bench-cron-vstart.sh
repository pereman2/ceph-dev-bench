#!/bin/bash

../src/stop.sh
FIO=~/fio/fio TESTDIR=$TESTDIR NRUNS=$NRUNS TEST=$TEST NEW_CLUSTER=1 FILL_CLUSTER=1 EXIT_ON_VSTART=1 COMPRESS_MODE=$COMPRESS_MODE ~/incerta-shared-testing/testrun-rewrite-4.sh