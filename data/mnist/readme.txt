The input folder contains the input file created from MNIST research dataset.  This dataset can be created from any machine learning or deep learning library.

We only provide this convenience. The scripts folder provide dimension estimation script for MNIST.

If one does not have access to sbatch, slrum setup simply extract the python command and execute.

Important:
Dimension estimation is resource intensive so don't stop the process before the singular value proxies start reducing.  This will over-estimate dimension of dataset.
