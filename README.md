# BM3D
BM3D denoising method implementation on Python

*BM3D.py* is an implementation based on my understanding of the method proposed by K. Dabov *et al.* in 2007. For more information, please have a visit at
[Image denoising by sparse 3D transform-domain collaborative filtering](http://www.cs.tut.fi/~foi/GCF-BM3D/)
and
[An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/).

## result
The output images of my code and official Matlab software are shown in PNG files for comparison and the PSNR is computed as the comparison criterion. The performance of my result is not as good as the official one since I have no idea whether I process some steps rightly in the method, like Wiener filtering.

## running time
The running time of the whole process is about 25 minutes which is much longer than that of the official code. There is still some work needed to reduce the computing complexity.

**Any suggestions on improving speed and final performance are welcome.** For that, contact with me

email: zhangchihao@zju.edu.cn

