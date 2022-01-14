from typing import Any
from unittest import result
from skimage import io
import numpy as np
import imageio
from sklearn.cluster import KMeans
from mpi4py import MPI



comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

img=io.imread("images/10.jpeg",as_gray=True)
centroids=np.array([[1000],[745],[240]])


if rank==0:

    chunks =np.array_split(img,size-1,0)
    for snode in range(0,size-1):
        comm.send(chunks[snode],dest=snode+1)
    finalimg=np.zeros((0, img.shape[1]))
    for snode in range(0,size-1):
        chunk=comm.recv(source=snode+1,tag=snode+1)
        finalimg=np.concatenate((finalimg,chunk))
    imageio.imwrite(("images/FinalImage.png"),finalimg)

else:
    img2=comm.recv(source=0,tag=MPI.ANY_TAG)
    img3=img2.reshape((-1,1))
    KmeansInstance=KMeans(n_clusters=3,init=centroids,max_iter=300,n_init=35,algorithm="elkan")
    model=KmeansInstance.fit(img3)
    Prediction=KmeansInstance.predict(img3)
    SegmentedImage=Prediction.reshape((img2.shape[0],img2.shape[1]))
    imageio.imwrite(("images/ChunkFrom"+str(rank)+".png"),SegmentedImage)
    comm.send(SegmentedImage,dest=0,tag=rank)




