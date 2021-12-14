import os

grid_size = 16
is_pinned = 1
stream_number = 4
enable_pinned_memory = 1

executable = 'streams'
output_folder = executable + '_output'

os.system("rm -fr ./{}".format(output_folder))
os.system("mkdir {}".format(output_folder))

while grid_size < 4096:
    enable_pinned_memory_str = "pinned" if enable_pinned_memory else "pageable"

    print ('[-] doing {} {} ...'.format(grid_size, enable_pinned_memory_str))
    os.system("nsys profile -o ./{}/{}.{}.{}.qdrep ./{} {} {} {}".format(output_folder, executable, grid_size, enable_pinned_memory_str, executable, grid_size, enable_pinned_memory, stream_number))
    os.system("./{} {} {} {} | awk 'NR==1{{print $NF}}' >> ./{}/{}_{}_totaltime".format(executable, grid_size, enable_pinned_memory, stream_number, output_folder, executable, enable_pinned_memory_str))

    if enable_pinned_memory == 0:
        grid_size <<= 1
    
    enable_pinned_memory = 0 if enable_pinned_memory else 1