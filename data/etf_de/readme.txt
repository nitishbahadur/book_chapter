The input folder contains the returns of SPDR ETF.  This dataset was created using prices from finance.yahoo.com, stricly for academic work.

The scripts folder provide dimension estimation of ETF on T using T-60 returns.  The scripts are only provided for convenience.

If one does not have access to sbatch, slrum setup simply extract the python command and execute.

We have provided scripts for all ETF's along with inputs and parameters.

Important:
Dimension estimation is resource intensive so don't stop the process before the singular value proxies start reducing.  This will over-estimate dimension of dataset.

Descripton of each dataset:
The file names follow <SPDR_ETF_Ticker>_returns.csv format.  For example, XLK_returns.csv is the input file for SPDR Technology ETF returns.