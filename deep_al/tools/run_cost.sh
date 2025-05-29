for label in POP
do
    for type in density clustered
    do
        for counties in 25 50 75 100 125 150 175 200
        do
            for radius in 10
            do
                initial_set_str="${type}_${counties}_counties_${radius}_radius"
                id_path="/home/libe2152/deep-al/usavars/population/sampled_points/${type}/IDs_${counties}_counties_10_radius_seed_42.pkl"
                for budget in $(seq 10 10 200)
                do
                    for method in random
                    do
                        for cost_func in pointwise_by_region
                        do
                            for region_assignment_str in state
                            do
                                region_assignment_path="/home/libe2152/deep-al/usavars/population/region_assignments/state.pkl"
                                for labeled_region_cost in 1
                                do
                                    for new_region_cost in 2
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
                                                --id-path=${id_path} \
                                                --cost_aware=True \
                                                --cost_func=${cost_func} \
                                                --region_assignment_path=${region_assignment_path} \
                                                --labeled_region_cost=${labeled_region_cost} \
                                                --new_region_cost=${new_region_cost}
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done