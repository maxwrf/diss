```bash
h5dump -d names  C57_CTX_G2CEPHYS3_TC11_DIV11_D.h5
````

```bash
ssh mw894@ssh.maths.cam.ac.uk "tar -czvf - /store/DAMTPEGLEN/mw894/slurm/Xu2011/*.h5" | tar -xzvf - -C .
````

```bash
rm /store/DAMTPEGLEN/mw894/slurm/*.out
```

``` bash
 scp mw894@ssh.maths.cam.ac.uk:"/store/DAMTPEGLEN/mw894/data/Charlesworth2015/ctx/group_*.h5" ./
 scp mw894@ssh.maths.cam.ac.uk:"/store/DAMTPEGLEN/mw894/data/Charlesworth2015/hcp/group_*.h5" ./
```

```bash
tmux: source /local/data/mphilcompbio/2022/mw894/gi2_a3_data/software/miniconda3/etc/profile.d/conda.sh
```

```bash
find .  -type f -name "*.subres" | wc -l
```
