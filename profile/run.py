import subprocess
import os
os.chdir("..")

path_to_profile = "./profile/"
path_to_exe = ".\\bin\\Release\\no\\cuda_path_tracer_with_stackless_kdtree.exe"
path_to_scene = "./scenes/cornell.txt"
path_to_obj = "./objects/pumpkin.obj"
for bound in [10, 20, 50, 100, 150, 200, 300]:
    for tri in range(1000, 11000, 1000):
        profile_file = path_to_profile + "no_partition.out"
        arg = path_to_exe + " " + path_to_scene + " " + str(bound) + " " + path_to_obj + " " + profile_file + " " + str(tri)
        print(arg)
        os.system(arg)

path_to_exe = ".\\bin\\Release\\par\\cuda_path_tracer_with_stackless_kdtree.exe"
for bound in [10, 20, 50, 100, 150, 200, 300]:
    for tri in range(1000, 11000, 1000):
        profile_file = path_to_profile + "partition.out"
        arg = path_to_exe + " " + path_to_scene + " " + str(bound) + " " + path_to_obj + " " + profile_file + " " + str(tri)
        print(arg)
        os.system(arg)

path_to_exe = ".\\bin\\Release\\naive\\cuda_path_tracer_with_stackless_kdtree.exe"
for bound in [100]:
    for tri in range(1000, 11000, 1000):
        profile_file = path_to_profile + "naive.out"
        arg = path_to_exe + " " + path_to_scene + " " + str(bound) + " " + path_to_obj + " " + profile_file + " " + str(tri)
        print(arg)
        os.system(arg)