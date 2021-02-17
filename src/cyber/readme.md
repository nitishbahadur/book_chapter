1.  Train model(s)
    - Execute command below with parameters updated as needed

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

2.  Determine \lambda to use
    a.  Edit paths to data in plot_ae_loss.py if needed
    b.  Execute python plot_ae_loss.py
    c.  Choose \lambda which provides lowest MSE

3.  Generate AE only scree plots
    a.  Edit generate_ae_scree_data if needed
    b.  Execute generate_ae_scree_data
    c.  Edit create_scree_plot.py if needed
    c.  Execute python create_scree_plot.py

4.  Determine SVP Threshold
    a.  Edit plot_ae_threshold to reference model chosen in (2)
    b.  Execute plot_ae_threshold
    c.  Choose SVP based on output from command above and comparison to scree plot

5.  Generate PCA results
    a.  Execute generate_pca_results

6.  Generate Combined results
    a.  Edit create_scree_plot.py if needed (to include PCA)
    b.  Execute python create_scree_plot.py
    c.  Edit generate_ae_results if needed
    d.  Execute generate_ae_results
    e.  Edit plot_dimensions_over_time.py if needed
    f.  Execute python plot_dimensions_over_time.py
