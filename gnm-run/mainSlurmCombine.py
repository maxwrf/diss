import numpy as np
from Slurm import Slurm

dSets = ["Charlesworth2015", "Hennig2011", "Demas2006"]


def main():
    cluster = False
    dSet = 0

    if cluster:
        inDirPath = "/store/DAMTPEGLEN/mw894/slurm/" + dSets[dSet]
    else:
        inDirPath = "/Users/maxwuerfek/code/diss/slurm/" + dSets[dSet]

    Kall, paramSpace, groupIds, filesRead = Slurm.combineResFiles(inDirPath)
    print("Files read: " + str(filesRead))
    Slurm.writeGroupsHD5(Kall, paramSpace, groupIds, inDirPath)
    print("Groups written:" + str(len(np.unique(groupIds))))


if __name__ == "__main__":
    main()
