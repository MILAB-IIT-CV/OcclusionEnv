import shutil
import os
import random
import os.path as osp

if __name__=='__main__':
    root = './Dataset/'
    oroot = './Dataset2/'

    os.makedirs(oroot, exist_ok=True)

    dirs = os.listdir(root)[1:]

    for dir in dirs:
        split = "train" if random.random() > 0.2 else "val"
        dst = osp.join(oroot, split)
        dst = osp.join(dst, dir)
        src = osp.join(root, dir)
        shutil.move(src, dst)

