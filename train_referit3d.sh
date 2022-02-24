cd referit3d/scripts
scanfile=keep_all_points_00_view_with_global_scan_alignment.pkl ## keep_all_points_with_global_scan_alignment if include Sr3D
python train_referit3d.py --patience 100 --max-train-epochs 100 \
                          --init-lr 1e-4 --batch-size 16 \
                          --transformer \
                          --model mmt_referIt3DNet -scannet-file $scanfile \
                          -referit3D-file $nr3dfile_csv --log-dir log/$exp_id \
                          --n-workers 2 --gpu 0 --unit-sphere-norm True \
                          --feat2d clsvecROI --context_2d unaligned \
                          --mmt_mask train2d --warmup