The input folder contains the input file created for estimating dimension of synthetic polynomial.  The synthetic polynomial mimics a random taylor series where the 
function contains
1.) 3rd degree polynomial in x, y, z and cross terms
2.) 2nd degree polynomial in x, y, z and cross terms
3.) 1st degree expressions in x, y, z and cross terms

The coefficients of the terms are random.  We call it synthetic polynomial because the expression was created using 3 independent variables: x, y, and z.

The scripts folder provide dimension estimation script for convenience.

If one does not have access to sbatch, slrum setup simply extract the python command and execute.

Important:
Dimension estimation is resource intensive so don't stop the process before the singular value proxies start reducing.  This will over-estimate dimension of dataset.
