import os

grid_size = 16
is_pinned = 1


os.system("rm -fr ./e1_output")
os.system("mkdir e1_output")

while grid_size < 4096:
    print ('[-] doing {} {} ...'.format(grid_size, "pinned"))
    os.system("nsys profile -o ./e1_output/e1.{}.{}.qdrep ./e1 {} {}".format(grid_size, "pinned", grid_size, 1))
    os.system("./e1 {} {} >> ./e1_output/e1_{}_totaltime".format(grid_size, 1, "pinned"))

    print ('[-] doing {} '.format(grid_size, "pageable"))
    os.system("nsys profile -o ./e1_output/e1.{}.{}.qdrep ./e1 {} {}".format(grid_size, "pageable", grid_size, 0))
    os.system("./e1 {} {} >> ./e1_output/e1_{}_totaltime".format(grid_size, 1, "pageable"))

    grid_size <<= 1