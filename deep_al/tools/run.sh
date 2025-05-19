for label in TC
do
    for id_path in \
        /home/libe2152/deep-al/usavars/sampled_points_treecover/IDs_clustered_500_counties_10_radius_seed_42.pkl \
        /home/libe2152/deep-al/usavars/sampled_points_treecover/IDs_density_500_counties_10_radius_seed_42.pkl
    do
        for budget in 100 200 300 400 500 1000
        do
            for method in random typiclust inversetypiclust
            do
                if [[ "$method" == "ensemble_variance" ]]; then
                    CUDA_VISIBLE_DEVICES=0 python ensemble_al.py \
                        --cfg=../configs/usavars/al/RIDGE_${label}_ENS.yaml \
                        --al=ensemble_variance \
                        --budget=${budget} \
                        --max_iter=1 \
                        --exp-name=USAVARS_${label}_AL_${method^^}_BUDGET_${budget}_IDPATH_$(basename $id_path) \
                        --id-path=${id_path} \
                        --seed=1
                else
                    for seed in 1 42 123 456 789
                    do
                        CUDA_VISIBLE_DEVICES=0 python train_al.py \
                            --cfg=../configs/usavars/al/RIDGE_${label}.yaml \
                            --al=${method} \
                            --budget=${budget} \
                            --max_iter=1 \
                            --exp-name=USAVARS_${label}_AL_${method^^}_BUDGET_${budget}_SEED_${seed}_IDPATH_$(basename $id_path) \
                            --seed=${seed} \
                            --id-path=${id_path}
                    done
                fi
            done
        done
    done
done
