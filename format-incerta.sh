#!/bin/bash

# usage ~/tester/format-incerta.sh result_folders* output_folder
# example:
#   ~/tester/format-incerta.sh ~/results/main-none ~/results/hybrid-alloc-igor_btree2/ ~/results/hybrid-alloc-igor-avl/ ~/tester

set -x

DIRS=($@)
DIRS=${DIRS[@]:0:$(($#-1))}
OUT=${!#}
TEMPS=()
for dir in $DIRS
do
    TEMP=$(mktemp -d)
    for file in $(ls $dir/*.fio)
    do
        name=$(basename $file .fio)
        sed -n '/^{/,/^}/p' $file > $TEMP/$name-fio.json
    done
    TEMPS+=($TEMP)
done

echo $DIRS 
echo $OUT
echo ${TEMPS[@]}

#files_to_exclude="22"
files_to_exclude="\-(8|9|10|11|12|13|14|15|16|17|18|19)\-"
#files_to_exclude="\-(16|17|18|19)\-"
# files_to_exclude="16|17|18|19"
# python3.11 ~/sshmount/tester/benchmarker.py --outplot $OUT/randwrite --testname randwrite compare --type fio empty --group $(ls -d $TEMP/*.json | grep randwrite | grep -v -E "${files_to_exclude}" | xargs) --group $(ls -d $TEMP2/*.json | grep randwrite | grep -v -E "${files_to_exclude}" | xargs) --group-names before after

group_names=""
for dir in $DIRS
do
  group_names="$group_names $(basename $dir)"
done

fio_file_names=(randread randwrite randrw rw allrw)
# fio_file_names=(allrw)

build_chart() {
  name=$1 # name corresponds to randread, randwrite, randrw, rw, etc.
  parse_name=$name
  # allrw corresponds to everything
  if [ "$parse_name" == "allrw" ]; then
    parse_name="randread|randwrite|randrw|rw"
  fi

  bench_groups=""
  for TEMP in ${TEMPS[@]}
  do
    echo $TEMP
    bench_groups="$bench_groups --group $(ls -d $TEMP/*.json | grep -E $parse_name | grep -v -E "${files_to_exclude}" | xargs)"
  done
  echo bench_groups: $bench_groups

  python3.11 ~/tester/benchmarker.py \
    --outplot $OUT/$name \
    --testname $name compare \
    --type fio empty \
    --group-names $group_names \
    $bench_groups 
}

build_chart_resources() {
  name="resources"
  files=""
  for dir in $DIRS
  do
      files="$files $dir/main.json"
  done

  echo bench files: $bench_groups

  python3.11 ~/tester/benchmarker.py \
    --outplot $OUT/$name \
    --testname $name compare \
    --type bench \
    $files
}

build_chart_resources
for fio_file_name in ${fio_file_names[@]}
do
  build_chart $fio_file_name
done

