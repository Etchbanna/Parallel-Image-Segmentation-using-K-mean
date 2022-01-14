from skimage import io
import numpy as np
import imageio
from sklearn.cluster import KMeans
from mpi4py import MPI


comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
img=io.imread("testimage.jpeg",as_gray=False)


if rank==0:

    chunks =np.array_split(img,size-1,0)
    for snode in range(0,size-1):
        comm.send(chunks[snode],dest=snode+1)
else:
    img2=comm.recv(source=0)
    img3=img2.reshape((-1,3))
    kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=42)
    model=kmeans.fit(img3)
    predicted_values=kmeans.predict(img3)

    segm_image=predicted_values.reshape((img2.shape[0],img2.shape[1]))
    imageio.imwrite(("Newimg"+str(rank)+".png"),segm_image)




