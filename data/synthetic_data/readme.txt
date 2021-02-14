The input folder contains the input file created for estimating dimension of synthetic dataset.

The file C_1586038282841_dim_5.npy has dimension 5. We use non-linear transformation and create D_1586038282841_dim_20.npy, which has linear dimension of 20.

We use Autoencoder to estimate dimension of D_1586038282841_dim_20.npy.  Since the intrinsic dimension of dataset is 5 we should get 5 even after non-linear transformations.

The scripts folder provide dimension estimation script for convenience.

If one does not have access to sbatch, slrum setup simply extract the python command and execute.

Important:
Dimension estimation is resource intensive so don't stop the process before the singular value proxies start reducing.  This will over-estimate dimension of dataset.
