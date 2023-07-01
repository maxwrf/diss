import os
import numpy as np
import h5py


class Slurm():
    @staticmethod
    def combineResFiles(inDirPath):
        Kall = []  # List of 4-dimensional lists
        groupIds = []
        filesRead = 0
        for filename in os.listdir(inDirPath):
            if os.path.isfile(os.path.join(inDirPath, filename)):
                if filename.split(".")[-1] == "res":
                    # Open the file
                    with open(os.path.join(inDirPath, filename), "rb") as file:
                        # Read in group Id size and resize
                        groupIdSize = int.from_bytes(
                            file.read(8), byteorder="little")

                        # Read in Kall dimensions and resize the array
                        size1Kall = int.from_bytes(
                            file.read(8), byteorder="little")
                        size2Kall = int.from_bytes(
                            file.read(8), byteorder="little")
                        size3Kall = int.from_bytes(
                            file.read(8), byteorder="little")

                        # Read in all param dimensions and resize the array
                        size1paramSpace = int.from_bytes(
                            file.read(8), byteorder="little")
                        size2paramSpace = int.from_bytes(
                            file.read(8), byteorder="little")

                        groupId = file.read(groupIdSize).decode("utf-8")

                        KallSample = np.fromfile(file, dtype=np.float64, count=size1Kall * size2Kall * size3Kall).reshape(
                            (size1Kall, size2Kall, size3Kall))

                        paramSpace = np.fromfile(file, dtype=np.float64).reshape(
                            (size1paramSpace, size2paramSpace))

                        Kall.append(KallSample)
                        groupIds.append(groupId)
                        filesRead += 1

        return np.array(Kall), paramSpace, np.array(groupIds), filesRead

    @staticmethod
    def writeGroupsHD5(Kall, paramSpace, groupIds, outDirPath):
        uniqueGroupIds = np.unique(groupIds)

        for groupId in uniqueGroupIds:
            indices = np.where(groupIds == groupId)
            KallGroup = Kall[indices]
            outFile = outDirPath + "/" + groupId + ".h5"
            with h5py.File(outFile, "w") as file:
                file.create_dataset("Kall", data=KallGroup)
                file.create_dataset("paramSpace", data=paramSpace)
                file.create_dataset("groupId", data=np.array(
                    [groupId.encode('utf-8')]))
