for label in POP
do
    for type in clustered density
    do
        for counties in 25 50 75 100 125 150 175 200
        do
            for radius in 10
            do
                initial_set_str="${type}_${counties}_counties_${radius}_radius"
                id_path="/home/libe2152/deep-al/usavars/population/sampled_points/${type}/IDs_${counties}_counties_10_radius_seed_42.pkl"

                for group_type in state
                do
                    if [ "$group_type" = "nlcd" ]; then
                        group_assignment_path="/home/libe2152/deep-al/usavars/population/cluster_assignments/NLCD_percentages_cluster_assignment.pkl"
                    elif [ "$group_type" = "state" ]; then
                        group_assignment_path="/home/libe2152/deep-al/usavars/population/region_assignments/state.pkl"
                    else
                        echo "Unknown group_type: $group_type"
                        continue
                    fi

                    echo "Using group_type: $group_type"
                    echo "Path: $group_assignment_path"

                    for budget in $(seq 10 10 100)
                    do
                        for method in representative
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
                                    --group_assignment_path=${group_assignment_path} \
                                    --group_type=${group_type}
                            done
                        done
                    done
                done
            done
        done
    done
done