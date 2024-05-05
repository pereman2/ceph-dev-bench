from typing import Any, Dict, List
import time
import sys
import psutil
import itertools
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import subprocess
import os
import json
import pandas as pd
from threading import Thread, Lock
COLORS = ["#f43f5e", "#8b5cf6", "#0ea5e9", "#059669", "#65a30d", "#f59e0b"]

class ProcessSample:
    def __init__(self, cpu_usage: float, memory_bytes: float, bluestore_data: Dict[str, Any], sample_time: float) -> None:
        self.cpu_usage = cpu_usage
        self.memory_bytes = memory_bytes
        self.sample_time = sample_time
        self.bluestore_data = bluestore_data


B_UNITS = {
    'B': 1,
    'KB': 1024,
    'MB': 1024*1024,
    'GB': 1024*1024*1024,
    'TB': 1024*1024*1024*1024,
}
S_UNITS = {
    's': 1,
    'ms': 1_000,
    'us': 1_000_000,
    'ns': 1_000_000_000,
}

def transform_bytes(value, unit):
    if value / B_UNITS['TB'] > 1:
        return value / B_UNITS['TB'], 'TB'
    elif value / B_UNITS['GB'] > 1:
        return value / B_UNITS['GB'], 'TB'
    elif value / B_UNITS['MB'] > 1:
        return value / B_UNITS['MB'], 'TB'
    elif value / B_UNITS['MB'] > 1:
        return value / B_UNITS['MB'], 'TB'
        
    return value, unit

def print_significance(ys, name, labels):
    from scipy import stats
    seen = {}
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            key = (i,j)
            if key in seen:
                continue
            key = (j,i)
            if key in seen:
                continue
            seen[key] = 1
            try:
                t_stat, p_value = stats.ttest_rel(ys[i], ys[j])
            except:
                continue

            # Check if the difference is statistically significant
            alpha = 0.05  # significance level
            if p_value < alpha:
                print(f"{name}: The difference is statistically significant between {labels[i]} and {labels[j]}.")
            else:
                print(f"{name}: There is no significant difference between {labels[i]} and {labels[j]}.")

def get_osd_pids():
    pids = []
    for proc in psutil.process_iter():
        if 'ceph-osd' in proc.name() and proc.username() == os.getlogin():
            pids.append(proc.pid)
    return pids

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class Benchmarker:
    def __init__(self, args):
        self.args = args
        self.finished_bench = False
        self.warmup_completed = False
        self.iteration = 0

    def benchmark_run_and_wait(self):
        # propagate env variables
        env = {k: v for k,v in os.environ.items()}
        env['TESTNAME'] = f'{self.args.testname}-{self.iteration}'
        bench_process = subprocess.Popen(f'{self.args.bench}'.split(' '), env=env)
        bench_process.wait()
        self.iteration += 1

    def run_bench(self, samples, iterations, warmup_iterations):
        print('Running warmup')
        for iteration in range(warmup_iterations):
            self.benchmark_run_and_wait()
        self.warmup_completed = True

        print('Running benchmark')
        for sample in range(samples):
            for iteration in range(iterations):
                self.benchmark_run_and_wait()

    def bench_entrypoint(self):
        self.run_bench(self.args.samples, self.args.iterations, self.args.warmup_iterations)
        self.finished_bench = True

    def run(self):
        if self.args.new_cluster:
            subprocess.run(f'../src/stop.sh'.split(' '))
            dir = os.path.dirname(__file__)
            do_vstart = os.path.join(dir, 'do_vstart.sh')
            if self.args.vstart:
                do_vstart = os.path.join(dir, self.args.vstart)
            subprocess.run(do_vstart)

            subprocess.run('bin/ceph osd pool create test'.split(' '))
            subprocess.run('bin/ceph osd pool set test pg_autoscale_mode off'.split(' '))
            subprocess.run(f'bin/ceph osd pool set test pg_num {self.args.pgs}'.split(' '))

        samples_per_process: Dict[int, List[ProcessSample]] = {}
        processes: List[psutil.Process] = []
        pids = get_osd_pids()
        for pid in pids:
            process = psutil.Process(int(pid))
            processes.append(process)
            process.cpu_percent()
            samples_per_process[process.pid] = []



        freq = self.args.freq
        start = time.time()
        period = self.args.period
        print(self.args.bench)
        run_bench = self.args.bench != ""
        if run_bench:
            bench_thread = Thread(target=self.bench_entrypoint)
            bench_thread.start()


        while run_bench and not self.warmup_completed:
            time.sleep(freq)

        names = {}
        for process in processes:
            pid = process.pid
            with open(f'/proc/{pid}/cmdline', 'r') as f:
                cmdline = f.read().split('\0')
                try:
                    i = cmdline.index('-i')
                except:
                    i = 1111
                name = f'osd.{cmdline[i+1]}'
                names[process.pid] = name

        if not run_bench:
            print(f'Collecting data for {period}s')
        dump_freq_time = time.time()
        print('start bench')
        while (run_bench and not self.finished_bench) or (not run_bench and time.time() - start < period):
            time.sleep(freq)
            do_dump = False
            if time.time() - dump_freq_time >= self.args.dump_freq:
                do_dump = True
            sample_time = (time.time_ns() - (start*1000000000)) / 1000000
            for process in processes:
                bluestore_data = {}
                if do_dump:
                    dump_freq_time = time.time()
                    result = subprocess.run(f'bin/ceph tell {names[process.pid]} perf dump'.split(' '), stdout=subprocess.PIPE, stderr=None, text=True)
                    dump = json.loads(result.stdout)
                    keys_to_save = ["bluestore", "bluestore-pricache", "bluestore-pricache:data", "bluestore-pricache:kv", "bluestore-pricache:kv_onode", "bluestore-pricache:meta"]

                    for key in keys_to_save:
                        bluestore_data[key] = dump[key]

                cpu_usage = process.cpu_percent()
                mem_bytes = process.memory_info().rss
                sample = ProcessSample(cpu_usage, mem_bytes, bluestore_data, sample_time)
                samples_per_process[process.pid].append(sample)
            
        
        print('end bench')
        if run_bench:
            bench_thread.join()


        # save data to compare with other runs
        save_data = {}

        for process in processes:
            name = f'{process.name()}.{process.pid}'
            save_data[name] = {}


        for process in processes:
            x = [sample.sample_time for sample in samples_per_process[process.pid]]
            ycpu = [sample.cpu_usage for sample in samples_per_process[process.pid]]
            # use MB
            ymem = [sample.memory_bytes/1024/1024 for sample in samples_per_process[process.pid]]
            bluestore_data = [sample.bluestore_data for sample in samples_per_process[process.pid]]

            name = f'{process.name()}.{process.pid}'
            save_data[name]['cpu'] = {}
            save_data[name]['cpu']['x'] = x
            save_data[name]['cpu']['y'] = ycpu
            save_data[name]['cpu']['stddev'] = np.std(ycpu)
            save_data[name]['cpu']['mean'] = np.mean(ycpu)
            save_data[name]['cpu']['median'] = np.median(ycpu)

            save_data[name]['mem'] = {}
            save_data[name]['mem']['x'] = x
            save_data[name]['mem']['y'] = ymem
            save_data[name]['mem']['stddev'] = np.std(ymem)
            save_data[name]['mem']['mean'] = np.mean(ymem)
            save_data[name]['mem']['median'] = np.median(ymem)

            x = [sample.sample_time for sample in samples_per_process[process.pid]]
            save_data[name]['bluestore_data'] = {}
            save_data[name]['bluestore_data']['x'] = x
            save_data[name]['bluestore_data']['y'] = bluestore_data

        now = int(time.time())

        file_path = f'{now}.json'
        if self.args.outdata:
            file_path = self.args.outdata
        with open(file_path, '+w') as f:
            json.dump(save_data, f)
            f.flush()

        files = [f'{now}.json']
        if os.path.exists('previous.json'):
            files.append('previous.json')

        with open(f'previous.json', '+w') as f:
            json.dump(save_data, f)
            f.flush()

    def compare_perfdumps(self, files=[]):
        keys = {
            'buffers': {'unit': 'N', 'division': 1},
            'buffer_bytes': {'unit': 'MB', 'division': 1024*1024},
            'buffer_hit_bytes': {'unit': 'MB', 'division': 1024*1024},
            'buffer_miss_bytes': {'unit': 'MB', 'division': 1024*1024},
            "write_big":{'unit': 'MB', 'division': 1024*1024},
            "write_big_bytes":{'unit': 'MB', 'division': 1024*1024},
            "write_big_blobs":{'unit': 'N', 'division': 1},
            "write_big_deferred":{'unit': 'N', 'division': 1},
            "write_small":{'unit': 'N', 'division': 1},
            "write_small_bytes":{'unit': 'MB', 'division': 1024*1024},
            "write_small_unused":{'unit': 'N', 'division': 1},
            "write_small_pre_read":{'unit': 'N', 'division': 1},
            "write_pad_bytes":{'unit': 'MB', 'division': 1024*1024},
            "write_penalty_read_ops":{'unit': 'N', 'division': 1},
            "write_new":{'unit': 'N', 'division': 1},
            "issued_deferred_writes":{'unit': 'N', 'division': 1},
            "issued_deferred_write_bytes":{'unit': 'MB', 'division': 1024*1024},
            "submitted_deferred_writes":{'unit': 'N', 'division': 1},
            "submitted_deferred_write_bytes":{'unit': 'MB', 'division': 1024*1024},
            "write_big_skipped_blobs":{'unit': 'N', 'division': 1},
            "write_big_skipped_bytes":{'unit': 'MB', 'division': 1024*1024},
            "write_small_skipped":{'unit': 'N', 'division': 1},
            "write_small_skipped_bytes":{'unit': 'MB', 'division': 1024*1024},
            "compressed":{'unit': 'MB', 'division': 1024*1024},
            "compressed_allocated":{'unit': 'MB', 'division': 1024*1024},
            "compressed_original":{'unit': 'MB', 'division': 1024*1024},
            "compress_success_count":{'unit': 'N', 'division': 1},
            "compress_rejected_count":{'unit': 'N', 'division': 1},
            "onodes":{'unit': 'N', 'division': 1},
            "onodes_pinned":{'unit': 'N', 'division': 1},
            "onode_hits":{'unit': 'N', 'division': 1},
            "onode_misses":{'unit': 'N', 'division': 1},
            "onode_shard_hits":{'unit': 'N', 'division': 1},
            "onode_shard_misses":{'unit': 'N', 'division': 1},
            "onode_extents":{'unit': 'N', 'division': 1},
            "onode_blobs":{'unit': 'N', 'division': 1},
            "onode_reshard":{'unit': 'N', 'division': 1},
            "blob_split":{'unit': 'N', 'division': 1},
            "extent_compress":{'unit': 'N', 'division': 1},
            "gc_merged":{'unit': 'N', 'division': 1},
            "omap_iterator_count":{'unit': 'N', 'division': 1},
            "omap_rmkeys_count":{'unit': 'N', 'division': 1},
            "omap_rmkey_range_count":{'unit': 'N', 'division': 1},
            "slow_aio_wait_count":{'unit': 'N', 'division': 1},
            "slow_committed_kv_count":{'unit': 'N', 'division': 1},
            "slow_read_onode_meta_count":{'unit': 'N', 'division': 1},
            "slow_read_wait_aio_count":{'unit': 'N', 'division': 1},
            "allocated":{'unit': 'MB', 'division': 1024*1024},
            "stored":{'unit': 'MB', 'division': 1024*1024},
            "fragmentation_micros":{'unit': 'MB', 'division': 1024*1024},
            "alloc_unit":{'unit': 'MB', 'division': 1024*1024},
            "txc_count": {'unit': 'N', 'division': 1},
            "read_eio": {'unit': 'N', 'division': 1},

            "state_prepare_lat": {'unit': 'μs', 'division': 0.001},
            "state_aio_wait_lat": {'unit': 'μs', 'division': 0.001},
            "state_io_done_lat": {'unit': 'μs', 'division': 0.001},
            "state_kv_queued_lat": {'unit': 'μs', 'division': 0.001},
            "state_kv_commiting_lat": {'unit': 'μs', 'division': 0.001},
            "state_kv_done_lat": {'unit': 'μs', 'division': 0.001},
            "state_finishing_lat": {'unit': 'μs', 'division': 0.001},
            "state_done_lat": {'unit': 'μs', 'division': 0.001},
            "state_deferred_queued_lat": {'unit': 'μs', 'division': 0.001},
            "state_deferred_aio_wait_lat": {'unit': 'μs', 'division': 0.001},
            "state_deferred_cleanup_lat": {'unit': 'μs', 'division': 0.001},
            "txc_commit_lat": {'unit': 'μs', 'division': 0.001},
            "txc_throttle_lat": {'unit': 'μs', 'division': 0.001},
            "kv_flush_lat": {'unit': 'μs', 'division': 0.001},
            "kv_commit_lat": {'unit': 'μs', 'division': 0.001},
            "kv_sync_lat": {'unit': 'μs', 'division': 0.001},
            "kv_final_lat": {'unit': 'μs', 'division': 0.001},
            "txc_submit_lat": {'unit': 'μs', 'division': 0.001},
            "read_onode_meta_lat": {'unit': 'μs', 'division': 0.001},
            "read_wait_aio_lat": {'unit': 'μs', 'division': 0.001},
            "csum_lat": {'unit': 'μs', 'division': 0.001},
            "reads_with_retries": {'unit': 'μs', 'division': 0.001},
            "read_lat": {'unit': 'μs', 'division': 0.001},
            "omap_seek_to_first_lat": {'unit': 'μs', 'division': 0.001},
            "omap_upper_bound_lat": {'unit': 'μs', 'division': 0.001},
            "omap_lower_bound_lat": {'unit': 'μs', 'division': 0.001},
            "omap_next_lat": {'unit': 'μs', 'division': 0.001},
            "omap_get_keys_lat": {'unit': 'μs', 'division': 0.001},
            "omap_get_values_lat": {'unit': 'μs', 'division': 0.001},
            "omap_clear_lat": {'unit': 'μs', 'division': 0.001},
            "clist_lat": {'unit': 'μs', 'division': 0.001},
            "remove_lat": {'unit': 'μs', 'division': 0.001},
            "truncate_lat": {'unit': 'μs', 'division': 0.001},
            "decompress_lat": {'unit': 'μs', 'division': 0.001},
            "compress_lat": {'unit': 'μs', 'division': 0.001},
        }
        simple_keys = {}

        for key in self.args.perf_counters:
            if key not in keys:
                print(f"{key} not not found, valid keys are {keys.keys()}")
                import sys
                sys.exit(1)
            else:
                simple_keys[key] = keys[key]



        # files must be multiples of 2, dump before and after to compare
        num_files = len(files)
        rows = len(simple_keys) # set minimum to two so we can use two dimensional indexing
        columns = len(files)
        fig, ax = plt.subplots(rows, columns, figsize=(30,5*rows))
        def plot_data(row, col, ylabel, title, x, y):
            ax[row, col].set_title(f'{title}')
            ax[row, col].set_xlabel('Time')
            ax[row, col].set_ylabel(ylabel)

            ax[row, col].plot(x, y)

        def plot_stats(row, col, y, avgcounts=[]):
            _max = np.max(y)
            _min = np.min(y)
            _avg = np.average(y)
            _p95 = np.percentile(y, 95)
            _p90 = np.percentile(y, 90)
            try:
                ax[row, col].set_yticks(np.arange(_min, _max, int(int(_max)/25)))
            except:
                pass
            ax[row, col].axhline(_avg, label='avg', color='red', in_layout=True)
            ax[row, col].axhline(_p90, label='p90', color='#c676db', in_layout=True)
            ax[row, col].axhline(_p95, label='p95', color='#a533c1', in_layout=True)
            ax[row, col].annotate(f'Avg: {_avg:.2f}', xy=(0, _avg), xytext=(5, _avg + 5),
                          horizontalalignment='left', verticalalignment='bottom')

            ax[row, col].annotate(f'P90: {_p90:.2f}', xy=(0, _p90), xytext=(5, _p90 + 5),
                          horizontalalignment='left', verticalalignment='bottom')

            ax[row, col].annotate(f'P95: {_p95:.2f}', xy=(0, _p95), xytext=(5, _p95 + 5),
                          horizontalalignment='left', verticalalignment='bottom')
            if avgcounts:
                # draw avgcount
                stats_box = ""
                for i, avgcount in enumerate(avgcounts):
                    stats_box += f"avgcount {i+1}: {avgcount:.2f}\n"
                ax[row, col].text(1.02, 1.0, stats_box, transform=ax[row, col].transAxes,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax[row, col].legend(bbox_to_anchor=(1.0, 1), loc='upper left')

        for i in range(len(files)):
            file_path = files[i]

            data = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)

            row = 0
            for key in simple_keys:
                aggregate = []
                avgcounts = []
                for name in data.keys():
                    x_unformatted = data[name]['bluestore_data']['x']
                    x = []
                    y = []
                    avgcount = 0
                    do_avgcount = False
                    for yn, v in enumerate(data[name]['bluestore_data']['y']):
                        if 'bluestore' not in v:
                            continue
                        v = v['bluestore']
                        if key in v:
                            x.append(x_unformatted[yn])
                            if key.endswith('_lat'):
                                y.append(v[key]['avgtime']/simple_keys[key]['division'])
                                do_avgcount = True
                                avgcount = max(avgcount, v[key]['avgcount'])
                            else:
                                y.append(v[key]/simple_keys[key]['division'])
                    aggregate += y
                    plot_data(row, i, f'{simple_keys[key]["unit"]}', f'{file_path} {key} {simple_keys[key]["unit"]}', x, y)
                    if do_avgcount:
                        avgcounts.append(avgcount)
                plot_stats(row, i, aggregate, avgcounts)
                row += 1 




        plt.tight_layout()
        plt.savefig(f'{self.args.outplot}.{self.args.format}', format=self.args.format)

    def compare_fiolog(self, files=[]):
        num_files = len(files)
        rows = 5 # set minimum to two so we can use two dimensional indexing
        rows += 2 # lat and iops evolution
        columns = 2 # iops, bw, lat...
        fig, ax = plt.subplots(rows, columns, figsize=(10,20))

        groups = [[file] for file in files]
        group_names = [str(i) for i in range(len(groups))]
        if self.args.group:
            groups = self.args.group
            group_names = self.args.group_names
            assert len(groups) == len(group_names)

        x = np.arange(len(groups))
        number_of_files_per_group = []
        write_iops = []
        read_iops = [] # avg and errors
        read_runtime = [] # avg and errors
        write_runtime = [] # avg and errors
        total_ios = [] # avg and errors
        total_bytes = [] # avg and errors
        read_bw = []
        write_bw = []
        read_clat = []
        write_clat = []
        labels = group_names
        KB = 1024
        MB = KB*1024
        GB = MB*1024
        for g, group in enumerate(groups):
            group_write_bw = []
            group_write_bw_dev = []
            group_write_iops = []
            group_write_iops_dev = []
            group_write_runtime = []
            group_write_clat = []

            group_read_bw = []
            group_read_bw_dev = []
            group_read_iops = []
            group_read_iops_dev = []
            group_read_runtime = []
            group_read_clat = []

            group_total_ios = []
            group_total_bytes = []

            number_of_files_per_group.append(len(group))
            for file in group:
                file_path = file

                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        jobs = data['jobs']
                        ios = 0
                        tbytes = 0
                        if 'read' in jobs[0]:
                            group_read_iops.append(jobs[0]['read']['iops_mean'] / 1000)
                            group_read_iops_dev.append(jobs[0]['read']['iops_stddev'] / 1000)

                            group_read_bw.append(jobs[0]['read']['bw_mean'] / MB)
                            group_read_bw_dev.append(jobs[0]['read']['bw_dev'] / MB)

                            group_read_clat.append(jobs[0]['read']['clat_ns']['mean'] / 1000000)
                            group_read_runtime.append(jobs[0]['read']['runtime']/60)
                            ios += jobs[0]['read']['total_ios']
                            tbytes += jobs[0]['read']['io_bytes']

                        if 'write' in jobs[0]:
                            group_write_iops.append(jobs[0]['write']['iops_mean'] / 1000)
                            group_write_iops_dev.append(jobs[0]['write']['iops_stddev'] / 1000)

                            group_write_bw.append(jobs[0]['write']['bw_mean'] / MB)
                            group_write_bw_dev.append(jobs[0]['write']['bw_dev'] / MB)

                            group_write_clat.append(jobs[0]['write']['clat_ns']['mean'] / 1000000)
                            group_write_runtime.append(jobs[0]['write']['runtime']/60)
                            ios += jobs[0]['write']['total_ios']
                            tbytes += jobs[0]['write']['io_bytes']

                        group_total_ios.append(ios/1000)
                        group_total_bytes.append(tbytes/1024/1024/1024)
                

            read_bw.append(group_read_bw)
            read_iops.append(group_read_iops)
            read_clat.append(group_read_clat)
            read_runtime.append(group_read_runtime)

            write_bw.append(group_write_bw)
            write_iops.append(group_write_iops)
            write_clat.append(group_write_clat)
            write_runtime.append(group_write_runtime)

            total_ios.append(group_total_ios)
            total_bytes.append(group_total_bytes)
        print(write_iops)
        print(read_iops)

        def plot_data_lines(row, col, title, file_path, x, y, labels):
            ax[row, col].set_xlabel('samples')
            ax[row, col].set_ylabel(title)
            print(x)
            print(y)
            for i, data in enumerate(y):
                newx = [a for a in range(len(data))]
                ax[row, col].plot(newx, data)
            for label in ax[row,col].get_xticklabels():
                label.set_rotation(90)

        def plot_data(row, col, title, file_path, x, y, labels):
            ax[row, col].set_xlabel('File')
            ax[row, col].set_ylabel(title)

            # ax[row, col].set_xticks(x)

            for label, yy in zip(labels, y):
                cv = np.std(yy)/np.mean(yy)
                print(f"coefficient of variation {title} {label}: {cv}")
                if cv > 1:
                    print(f"HIGH coefficient of variation {title} {label}: {cv}")
            box = ax[row, col].boxplot(y, patch_artist=True, labels=labels)
            for patch, color in zip(box['boxes'], COLORS):
                patch.set_facecolor(color)

            scatter_x = [np.full_like(data, i + 1) for i, data in enumerate(y)]
            scatter_x = np.concatenate(scatter_x)
            scatter_y = [data for i, data in enumerate(y)]
            ax[row, col].scatter(scatter_x, scatter_y, alpha=0.8, marker='o', zorder=10, color=COLORS[-1])

            # ax[row, col].legend()


                    

        print_significance(read_iops, 'read iops', labels)
        print_significance(write_iops, 'write iops', labels)

        plot_data_args = [
                (False, 0, 0, 'read kIOPS ', file_path, x, read_iops, labels),
                (False, 0, 1, 'write kIOPS', file_path, x, write_iops, labels),
                (False, 1, 0, 'read bandwidth MBs', file_path, x, read_bw, labels),
                (False, 1, 1, 'write bandwidth MBs', file_path, x, write_bw, labels),
                (False, 2, 0, 'read clat ms', file_path, x, read_clat, labels),
                (False, 2, 1, 'write clat ms', file_path, x, write_clat, labels),
                (False, 3, 0, 'read runtime min', file_path, x, read_runtime, labels),
                (False, 3, 1, 'write runtime min', file_path, x, write_runtime, labels),
                (False, 4, 0, 'total kIOS', file_path, x, total_ios, labels),
                (False, 4, 1, 'total GB', file_path, x, total_bytes, labels),
                (True, 5, 0, 'lines read kIOPS ', file_path, x, read_iops, labels),
                (True, 5, 1, 'lines write kIOPS', file_path, x, write_iops, labels),
                (True, 6, 0, 'lines read clat ms', file_path, x, read_clat, labels),
                (True, 6, 1, 'lines write clat ms', file_path, x, write_clat, labels),
        ]
        for args in plot_data_args:
            if not args[0]:
                plot_data(*args[1:])
            else:
                plot_data_lines(*args[1:])

        plt.tight_layout()
        plt.savefig(f'{self.args.outplot}.{self.args.format}', format=self.args.format)

        # save to excel
        csv_writer = pd.ExcelWriter(f'{self.args.outplot}.fio_stats.xlsx', engine='xlsxwriter')
        dataframes = []
        for r, w, rb, wb, rl, wl, tb, label in zip(read_iops, write_iops, read_bw, write_bw, read_clat, write_clat, total_bytes, labels):
            df = pd.DataFrame({'Read iops': r, 'Write iops': w, 'Read Bandwidth': rb, 'Write Bandwidth': wb, 'Read Latency': rl, 'Write Latency': wl, 'Total Bytes': tb})
            description = df.describe([.5, .75, .95, .99])
            dataframes.append((description, label))
            print(description)
            description.to_excel(csv_writer, sheet_name=label)
        for d1, d2 in itertools.combinations(dataframes, 2):
            comparison = ((d2[0] / d1[0]) - 1)
            styler = comparison.style.background_gradient().format(precision=2)
            sheet_name = f'{d2[1]} to {d1[1]}'
                # xsls sucks ass and doesn't let me making huge sheet names :(
            sheet_name = sheet_name[:31]
            styler.to_excel(csv_writer, sheet_name=sheet_name)
        csv_writer.close()


    def compare_resources(self, files):
        num_files = len(files)
        rows = max(num_files, 2) # set minimum to two so we can use two dimensional indexing
        columns = 2 # mem, cpu
        fig, ax = plt.subplots(rows, columns, figsize=(30,20))

        compares =[]
        for i in range(len(files)):
            file_path = files[i]

            data = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)

            cpu_aggregate: np.ndarray = np.array([])
            mem_aggregate: np.ndarray = np.array([])
            for name in data.keys():
                x = data[name]['cpu']['x']
                ycpu = np.array(data[name]['cpu']['y'])
                ymem = np.array(data[name]['mem']['y'])

                cpu_aggregate = np.append(cpu_aggregate, ycpu)
                mem_aggregate = np.append(mem_aggregate, ymem)


                def plot_data(row, col, title, file_path, x, y):
                    ax[row, col].set_title(f'{title} {file_path}')
                    ax[row, col].set_xlabel('Time')
                    ax[row, col].set_ylabel(title)

                    ax[row, col].plot(x, smooth(y, 10), dash_joinstyle="round", color='#507baf')


                plot_data(i, 0, 'Cpu usage', file_path, x, ycpu)
                plot_data(i, 1, 'Memory (MB)', file_path, x, ymem)

            def plot_stats(row, col, y):
                _max = np.max(y)
                _min = np.min(y)
                _avg = np.average(y)
                _p95 = np.percentile(y, 95)
                _p90 = np.percentile(y, 90)
                try:
                    ax[row, col].set_yticks(np.arange(_min, _max, int(_max/50)))
                except:
                    ax[row, col].set_yticks(y)
                ax[row, col].axhline(_avg, label='avg', color='red', in_layout=True)
                ax[row, col].axhline(_p90, label='p90', color='#c676db', in_layout=True)
                ax[row, col].axhline(_p95, label='p95', color='#a533c1', in_layout=True)
                ax[row, col].annotate(f'Avg: {_avg:.2f}', xy=(0, _avg), xytext=(5, _avg + 5),
                            horizontalalignment='left', verticalalignment='bottom')

                ax[row, col].annotate(f'P90: {_p90:.2f}', xy=(0, _p90), xytext=(5, _p90 + 5),
                            horizontalalignment='left', verticalalignment='bottom')

                ax[row, col].annotate(f'P95: {_p95:.2f}', xy=(0, _p95), xytext=(5, _p95 + 5),
                            horizontalalignment='left', verticalalignment='bottom')
                ax[row, col].legend(bbox_to_anchor=(1.0, 1), loc='upper left')
            compares.append([cpu_aggregate, mem_aggregate])

            print(file_path, len(cpu_aggregate), len(mem_aggregate))
            plot_stats(i, 0, cpu_aggregate)
            plot_stats(i, 1, mem_aggregate)

        def print_stat_differences(compare1, compare2, f1, f2, title, statf, *args):
            diffcpu = round(((statf(compare1[0], *args) / statf(compare2[0], *args)) - 1) * 100, 2)
            diffmem = round(((statf(compare1[1], *args) / statf(compare2[1], *args)) - 1) * 100, 2)
            print(f'')
            print(f'Difference {title} --------------------------------')
            if diffcpu > 0:
                print(f'{f2} uses less cpu {diffcpu}% than {f1}')
            else:
                print(f'{f1} uses less cpu {diffcpu}% than {f2}')
            if diffmem > 0:
                print(f'{f2} uses less mem {diffmem}% than {f1}')
            else:
                print(f'{f1} uses less mem {diffmem}% than {f2}')
                

        print(compares)
        # print_significance([c[0].flatten() for c in compares], 'cpu', files)
        # print_significance([c[1].flatten() for c in compares], 'mem', files)

        seen_compares = {}
        for i, f1 in enumerate(files):
            for j, f2 in enumerate(files):
                key = (i,j)
                if key in seen_compares :
                    continue
                key = (j,i)
                if key in seen_compares :
                    continue
                if f1 == f2:
                    continue
                seen_compares[key] = 1
                print_stat_differences(compares[i], compares[j], f1, f2, 'average', np.average, None)
                print_stat_differences(compares[i], compares[j], f1, f2, 'p70', np.percentile, 70)
                print_stat_differences(compares[i], compares[j], f1, f2, 'p90', np.percentile, 90)
                print_stat_differences(compares[i], compares[j], f1, f2, 'p95', np.percentile, 95)
            
        plt.tight_layout()
        plt.savefig(f'{self.args.outplot}.{self.args.format}', format=self.args.format)

    def compare(self, files=[]):
        if not files:
            files = self.args.files
        if self.args.type == 'bench':
            self.compare_resources(files)
        elif self.args.type == 'fio':
            self.compare_fiolog(files)
        elif self.args.type == 'perfdump':
            self.compare_perfdumps(files)

    
    def system_determinism(self):
        write = False
        enable = False
        if self.args.enable == 'on' or self.args.enable == 'off':
            write = True
            if self.args.enable == 'on':
                enable = True
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'Intel' in cpuinfo:
                pass
            elif 'AMD' in cpuinfo:
                print("ERROR AMD not supported yet")
                sys.exit(1)
            else:
                print('ERROR: We are not Intel or AMD')
                sys.exit(1)

        # turbo boost
        if write:
            with open('/sys/devices/system/cpu/intel_pstate/no_turbo', 'w') as f:
                f.write('1' if enable else '0')
        else:
            with open('/sys/devices/system/cpu/intel_pstate/no_turbo', 'r') as f:
                print(f'turbo boost is enabled: {f.read().strip() == "0"}')

        cpu_dir = "/sys/devices/system/cpu"
        cpu_folders = os.listdir(cpu_dir)
        for cpu_folder in cpu_folders:
            if not cpu_folder.startswith('cpu') and not cpu_folder[3:].isdigit():
                continue

            # hyperthreading
            if not os.path.exists(f'{cpu_dir}/{cpu_folder}/online'):
                continue

            print(f'changing cpu {cpu_dir}/{cpu_folder}')
            if write:
                print("hyper threading wall, remove \"if False\" if you want to enable/disable hyperthreading")
                if False:
                    # siblings are always written in the same order, so it is supposedly to be correct to always disable everything but the first cpu
                    if enable:
                        if os.path.exists(f'{cpu_dir}/{cpu_folder}/topology/thread_siblings_list'):
                        # only write to right sibling
                            cpu_folder_siblings = []
                            with open(f'{cpu_dir}/{cpu_folder}/topology/thread_siblings_list', 'r') as f:
                                siblings = f.read().strip().split(',')
                                cpu_folder_siblings = siblings[1:]
                            for cpu_folder_sibling in cpu_folder_siblings:
                                print(f'sibling {cpu_folder_sibling}')
                                with open(f'{cpu_dir}/cpu{cpu_folder_sibling}/online', 'w') as f:
                                    f.write('0' if enable else '1')
                    else:
                        print(f'enabling hyperthreading for {cpu_folder}')
                        with open(f'{cpu_dir}/{cpu_folder}/online', 'w') as f:
                            f.write('0' if enable else '1')
            else:
                with open(f'{cpu_dir}/{cpu_folder}/online', 'r') as f:
                    print(f'cpu hyperthreading {cpu_folder:<6} is online: {f.read().strip() == "1"}')
            # scaling governor
            if write:
                print(f'{cpu_dir}/{cpu_folder}/cpufreq/scaling_governor')
                try:
                    with open(f'{cpu_dir}/{cpu_folder}/cpufreq/scaling_governor', 'w') as f:
                        value = "performance" if enable else "ondemand"
                        f.write(value)
                except:
                    print("cant write to scaling_governor in this system")
            else:
                with open(f'{cpu_dir}/{cpu_folder}/cpufreq/scaling_governor', 'r') as f:
                    print(f'cpu scaling governor {cpu_folder:<6}: {f.read()}')

        # address space layout randomization
        if write:
            with open('/proc/sys/kernel/randomize_va_space', 'w') as f:
                f.write('0' if enable else '2')
        else:
            with open('/proc/sys/kernel/randomize_va_space', 'r') as f:
                print(f'address space layout randomization: {f.read().strip() == "0"}')



        print(self.args.enable)
        print('inside')



def main():
    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('--period', type=int)
    parser.add_argument('--new-cluster', '-n', action='store_true', help="Set this flag if you want to create a new cluster with available parameters.")
    parser.add_argument('--pgs', type=int, default=128)
    parser.add_argument('--freq', type=int, default=0.1, help="Frequency of sampling")
    parser.add_argument('--dump-freq', type=int, default=5, help="Frequency of sampling perf dump")
    parser.add_argument('--samples', type=int, default=5, help="Number of samples")
    parser.add_argument('--iterations', type=int, default=5, help="Number of iterations per sample")
    parser.add_argument('--warmup-iterations', type=int, default=1, help="Number of iterations per sample")
    parser.add_argument('--bench', type=str, default="")
    parser.add_argument('--vstart', type=str, default="")
    parser.add_argument('--format', help="format of matplotlib plots saved image", type=str, default='png')
    parser.add_argument('--outplot', type=str, default="benchmark")
    parser.add_argument('--outdata', type=str, default="")
    parser.add_argument('--testname', type=str, required=True, help='used to set ENV TESTNAME')
    parser.set_defaults(func=Benchmarker.run)

    compare_subparser = parser.add_subparsers(help="compare results")
    compare_parser = compare_subparser.add_parser('compare', help="compare results")
    compare_parser.add_argument('files', help="Results to compare", type=str, nargs='+')
    compare_parser.add_argument('--type', help="File type to parse [bench,fio,perfdump]", type=str, default='bench')
    group_help = """
        Files to join stats, group like --group a b --group c d

        a and b will combine their stats for a bar
        c and c will combine their stats for another bar
    """
    compare_parser.add_argument('--group', help=group_help, action='append', nargs='+')
    compare_parser.add_argument('--group-names', help="names of each group sorted --group", type=str, nargs='*')
    compare_parser.add_argument('--fiolog', help="Parse log fio files", action='store_true')
    default_perfs = [
        "buffers",
        "buffer_bytes",
        "buffer_hit_bytes",
        "buffer_miss_bytes",
        "allocated",
        "stored",
        "state_done_lat",
        "compressed",
        "compressed_allocated",
        "compressed_original",
    ]
    compare_parser.add_argument('--perf-counters', help="Perf counters to use", type=str, nargs="*", default=default_perfs)
    compare_parser.set_defaults(func=Benchmarker.compare)

    # system_subparser = parser.add_subparsers(help="Disable/Enable system settings like Hyperthreading, turbo boost, etc") 
    system_parser = compare_subparser.add_parser('system_determinism', help="Disable/Enable system settings like Hyperthreading, turbo boost, etc")
    system_parser.add_argument('enable', help="Disable/Enable system features that can cause noise while benchmarking\n Values: on/off/status", type=str)
    system_parser.set_defaults(func=Benchmarker.system_determinism)

    args = parser.parse_args()
    print(args.__dict__)
    bench = Benchmarker(args)
    args.func(bench)


if __name__ == "__main__":
    main()
