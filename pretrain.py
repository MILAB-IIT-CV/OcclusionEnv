from dataset import OcclusionDataset

if __name__=='__main__':
    trainSet = OcclusionDataset("./DataSet/", "train")

    trainSet[0]