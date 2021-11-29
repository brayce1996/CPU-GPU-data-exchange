import os

grid_size = 16
is_pinned = 1

os.system("rm -fr ./e2_output")
os.system("mkdir e2_output")

while grid_size < 4096:
    print ('[-] doing {} {} ...'.format(grid_size, "pinned"))
    os.system("nsys profile -o ./e2_output/e2.{}.{}.qdrep ./e2 {} {} 4".format(grid_size, "pinned", grid_size, 1))
    os.system("./e2 {} {} 4 >> ./e2_output/e2_{}_totaltime".format(grid_size, 1, "pinned"))

    print ('[-] doing {} '.format(grid_size, "pageable"))
    os.system("nsys profile -o ./e2_output/e2.{}.{}.qdrep ./e2 {} {} 4".format(grid_size, "pageable", grid_size, 0))
    os.system("./e2 {} {} 4 >> ./e2_output/e2_{}_totaltime".format(grid_size, 1, "pageable"))

    grid_size <<= 1