import subprocess
import os
os.chdir("..")

path_to_profile = "./profile/"
path_to_exe = ".\\bin\\Release\\cuda_path_tracer_with_stackless_kdtree.exe"
path_to_scene = "./scenes/cornell.txt"
path_to_obj = "./objects/cessna.obj"
for bound in [50, 100, 150, 200, 300]:
    profile_file = path_to_profile + "no_partition.out"
    arg = path_to_exe + " " + path_to_scene + " " + str(bound) + " " + path_to_obj + " " + profile_file + " " + str(1000000)
    print(arg)
    os.system(arg)