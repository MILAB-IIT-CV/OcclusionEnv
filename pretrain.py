from dataset import OcclusionDataset

if __name__=='__main__':
    trainSet = OcclusionDataset("./Dataset/", "train")

    print(trainSet[0])