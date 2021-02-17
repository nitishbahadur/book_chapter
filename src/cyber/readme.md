# General Process Overview

##  Train model(s)
    * Execute command below with parameters updated as needed

    python cyber_approach_2.py \
        --tp 1min \
        --l1_reg 0.0001 \
        --num_epochs 400 \
        --batch_size 32 \
        --encoding_dim 25 \
        --layer1_dropout 0.01 \
        --layer2_dropout 0.01 \
        --layer3_dropout 0.01 \
        --dataset ./datasets/cicids2017/monday_benign_traffic/monday_benign_traffic.pkl \
        --output_dir ./output/20210206_ae_l1/ae_0.0001 \
        --plot_title 'CICIDS2017 Monday'

##  Determine \lambda to use
    *  Edit paths to data in plot_ae_loss.py if needed
    *  Execute python plot_ae_loss.py
    *  Choose \lambda which provides lowest MSE

##  Generate AE only scree plots
    *  Edit generate_ae_scree_data if needed
    *  Execute generate_ae_scree_data
    *  Edit create_scree_plot.py if needed
    *  Execute python create_scree_plot.py

##  Determine SVP Threshold
    *  Edit plot_ae_threshold to reference model chosen in (2)
    *  Execute plot_ae_threshold
    *  Choose SVP based on output from command above and comparison to scree plot

##  Generate PCA results
    *  Execute generate_pca_results

##  Generate Combined results
    *  Edit create_scree_plot.py if needed (to include PCA)
    *  Execute python create_scree_plot.py
    *  Edit generate_ae_results if needed
    *  Execute generate_ae_results
    *  Edit plot_dimensions_over_time.py if needed
    *  Execute python plot_dimensions_over_time.py
