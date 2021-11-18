# Ray tracing difussion curves

An Optix based implementation of A ray tracing approach to A Ray Tracing Approach to Diffusion Curves [Bowers, Leahey and Wang 2011] with several extensions.


The program can be compiled using CMake and is tested on the windows platform using Visual Studio, CUDA 11.3 and Optix 7.3

To use the program compile it using your favorite compiler and hope that it works. Then put the Xmls directory in the same directory as the executable and call the programm from the command line with the xml and the amount of rays per pixel. For example :

```
./OptixHello.exe xmls/arch.xml 128
```

In the params.hpp files there are several defines which can be used to flag that the diffusion curve xml is saved from the implementation of Orzan et al. [Orzan et. al. 2008], If the blur functionality should be used, if the randomization within each pixel should be used and the last option is for disabling the optix denoiser. These are all best left to true.

Another define sets the maximum ray trace depth the programm should allow untill it returns a miss. Optix only allows a maximum depth of 31. But unless the connects atribute is used it should not give a perfomance penalty if set higher than 0.

lastly the OptixHello.cpp file contains the initial values for the zoom, offset in x and y direction, the default weight weight exponent used if not defined, width of the diffusion curves, endcap size and lastly the mix ratio used for the denoiser. The defaults of these values should work for most images but optimizing these might give better results. 

This project was the result of my bachelor thesis which can be found here https://repository.tudelft.nl/islandora/object/uuid:3e8e5679-5e05-4989-81ac-0c5569614597?collection=education.
