import os

grid_size = 16
is_pinned = 1

executable = 'split_computation'
output_folder = executable + '_output'

os.system("rm -fr ./{}".format(output_folder))
os.system("mkdir {}".format(output_folder))

while grid_size < 4096:
    print ('[-] doing {} {} ...'.format(grid_size, "pinned"))
    enable_pinned_memory = 1
    os.system("nsys profile -o ./{}/{}.{}.{}.qdrep ./{} {} {}".format(output_folder, executable, grid_size, "pinned", executable, grid_size, enable_pinned_memory))
    os.system("./{} {} {} | awk '{{print $NF}}' >> ./{}/{}_{}_totaltime".format(executable, grid_size, enable_pinned_memory, output_folder, executable, "pinned"))


    print ('[-] doing {} '.format(grid_size, "pageable"))
    enable_pinned_memory = 0
    os.system("nsys profile -o ./{}/{}.{}.{}.qdrep ./{} {} {}".format(output_folder, executable, grid_size, "pageable", executable, grid_size, enable_pinned_memory))
    os.system("./{} {} {} | awk '{{print $NF}}' >> ./{}/{}_{}_totaltime".format(executable, grid_size, enable_pinned_memory, output_folder, executable, "pageable"))

    grid_size <<= 1