#!/bin/bash
mkdir -p g0 g1 g2 g3


FIO=~/fio/fio TESTDIR=$TESTDIR TEST=$TEST NEW_CLUSTER=0 NRUNS=$NRUNS FILL_CLUSTER=0 COMPRESS_MODE=$COMPRESS_MODE ~/incerta-shared-testing/testrun-rewrite-4.sh