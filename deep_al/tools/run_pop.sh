for label in POP
do
    for type in density clustered
    do
        for counties in 125 150 175 200 100 75 50 25
        do
            for radius in 10
            do
                initial_set_str="${type}_${counties}_counties_${radius}_radius"
                id_path="/home/libe2152/deep-al/usavars/sampled_points_population/${type}/IDs_${counties}_counties_10_radius_seed_42.pkl"
                for budget in $(seq 20 10 90)
                do
                    for method in random typiclust inversetypiclust
                    do
                        for seed in 1 42 123 456 789
                        do
                            CUDA_VISIBLE_DEVICES=0 python train_al.py \
                                --cfg=../configs/usavars/al/RIDGE_${label}.yaml \
                                --al=${method} \
                                --budget=${budget} \
                                --max_iter=1 \
                                --initial_set_str=${initial_set_str} \
                                --exp-name=USAVARS_${label}_AL_${method^^}_BUDGET_${budget}_SEED_${seed}_TYPE_${type^^}_COUNTIES_${counties}_RADIUS_${radius} \
                                --seed=${seed} \
                                --id-path=${id_path}
                        done
                    done
                done
            done
        done
    done
done
