from genericpath import isfile
import os

for file in os.path.listdir(os.curdir):
    if not isfile(file) or file.endswith(".npz"):
        print("Skipping {}".format(file))
        continue
