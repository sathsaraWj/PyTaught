How to run

Save the file pytaught.py.

(Optional) Make executable:

chmod +x mpytaught.py


Train on a CSV:

The CSV must include numeric columns; non-numeric columns are dropped by this simple loader.

Example (train an MLP):

python pytaught.py --csv housing.csv --target price --model mlp --hidden-sizes 128 64 --lr 0.001 --epochs 200 --batch-size 32 --save checkpoints/housing_model.npz


Example (linear regression):

python pytaught.py --csv housing.csv --target price --model linear --lr 0.01 --epochs 100


If you don't pass --csv, a synthetic dataset will be generated and trained on to demonstrate the pipeline.