import os
import csv

grid_size = 16
is_pinned = 1
enable_pinned_memory = 1

executable = 'normal'
output_folder = executable + '_output'

os.system("rm -fr ./{}".format(output_folder))
os.system("mkdir {}".format(output_folder))

fields = ["data_size", "total_time", "page_faults", "cache_misses", "llc_load_misses"]
pinned_rows = []
pageable_rows = []

while grid_size < 4096:

    enable_pinned_memory_str = "pinned" if enable_pinned_memory else "pageable"

    print ('[-] doing {} {} ...'.format(grid_size, enable_pinned_memory_str))
    os.system("nsys profile -o ./{}/{}.{}.{}.qdrep ./{} {} {}".format(output_folder, executable, grid_size, enable_pinned_memory_str, executable, grid_size, enable_pinned_memory))
    total_time = os.popen("./{} {} {} | awk 'NR==1{{print $NF}}'".format(executable, grid_size, enable_pinned_memory)).read().strip()
    page_faults = os.popen("perf stat -e faults ./{} {} {} 2>&1 >/dev/null | awk '{{if ($2 == \"faults\") {{print $1}} }}'".format(executable, grid_size, enable_pinned_memory)).read().strip()
    cache_misses = os.popen("perf stat -e cache-misses ./{} {} {} 2>&1 >/dev/null | awk '{{if ($2 == \"cache-misses\") {{print $1}} }}'".format(executable, grid_size, enable_pinned_memory)).read().strip()
    llc_load_misses = os.popen("perf stat -e LLC-load-misses ./{} {} {} 2>&1 >/dev/null | awk '{{if ($2 == \"LLC-load-misses\") {{print $1}} }}'".format(executable, grid_size, enable_pinned_memory)).read().strip()

    row = [grid_size, total_time, page_faults, cache_misses, llc_load_misses]

    if enable_pinned_memory == 0:
        pageable_rows.append(row)
        grid_size <<= 1
    else:
        pinned_rows.append(row)
    
    enable_pinned_memory = 0 if enable_pinned_memory else 1

print ('[-] writing results to csv ...')
with open('{}/{}_pageable.csv'.format(output_folder, executable), 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(pageable_rows)

with open('{}/{}_pinned.csv'.format(output_folder, executable), 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(pinned_rows)
