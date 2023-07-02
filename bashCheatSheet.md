```bash
h5dump -d names  C57_CTX_G2CEPHYS3_TC11_DIV11_D.h5
````

```bash
ssh mw894@ssh.maths.cam.ac.uk "tar -czvf - /store/DAMTPEGLEN/mw894/slurm/Xu2011/*.h5" | tar -xzvf - -C .
````

```bash
rm /store/DAMTPEGLEN/mw894/slurm/*.out
```