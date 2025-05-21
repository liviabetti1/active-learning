for label in TC
do
    for id_path in \
        /home/libe2152/deep-al/usavars/sampled_points_treecover/IDs_clustered_500_counties_10_radius_seed_42.pkl \
        /home/libe2152/deep-al/usavars/sampled_points_treecover/IDs_density_500_counties_10_radius_seed_42.pkl 
    do
        id_file=$(basename $id_path)
        id_file_no_ext="${id_file%.pkl}"  # Remove .pkl extension

        if [[ $id_file == *"clustered"* ]]; then
            strategy="clustered"
        elif [[ $id_file == *"density"* ]]; then
            strategy="density"
        else
            echo "Unknown strategy in $id_file"
            exit 1
        fi

        if [[ $id_file == *"100_counties"* ]]; then
            counties="100"
        elif [[ $id_file == *"500_counties"* ]]; then
            counties="500"
        else
            echo "Unknown county count in $id_file"
            exit 1
        fi

        seed=$(echo $id_file | grep -oP 'seed_\K[0-9]+')

        cost_file="cost_${strategy}_${counties}_counties_r1_10_r2_20_seed_${seed}_cost_1_vs_2.pkl"
        cost_path="/home/libe2152/deep-al/usavars/cost_treecover/${cost_file}"

        r_part=$(echo $cost_file | grep -oP 'r1_\K[0-9]+_r2_[0-9]+')
        cost_part=$(echo $cost_file | grep -oP 'cost_\K[0-9+_vs_0-9]+')

        for budget in 100 200 300 400 500 1000
        do
            for method in greedycost
            do
                for train_seed in 1 42 123 456 789
                do
                    CUDA_VISIBLE_DEVICES=0 python train_al.py \
                        --cfg=../configs/usavars/al/RIDGE_${label}.yaml \
                        --al=${method} \
                        --budget=${budget} \
                        --max_iter=1 \
                        --exp-name=USAVARS_${label}_AL_${method^^}_BUDGET_${budget}_SEED_${train_seed}_IDPATH_${id_file_no_ext}_R${r_part}_COST${cost_part} \
                        --seed=${train_seed} \
                        --id-path=${id_path} \
                        --cost_path=${cost_path}
                done
            done
        done
    done
done

for label in POP
do
    for id_path in \
        /home/libe2152/deep-al/usavars/sampled_points_population/IDs_clustered_500_counties_10_radius_seed_42.pkl \
        /home/libe2152/deep-al/usavars/sampled_points_population/IDs_density_500_counties_10_radius_seed_42.pkl 
    do
        id_file=$(basename $id_path)
        id_file_no_ext="${id_file%.pkl}"  # Remove .pkl extension

        if [[ $id_file == *"clustered"* ]]; then
            strategy="clustered"
        elif [[ $id_file == *"density"* ]]; then
            strategy="density"
        else
            echo "Unknown strategy in $id_file"
            exit 1
        fi

        if [[ $id_file == *"100_counties"* ]]; then
            counties="100"
        elif [[ $id_file == *"500_counties"* ]]; then
            counties="500"
        else
            echo "Unknown county count in $id_file"
            exit 1
        fi

        seed=$(echo $id_file | grep -oP 'seed_\K[0-9]+')

        cost_file="cost_${strategy}_${counties}_counties_r1_10_r2_20_seed_${seed}_cost_1_vs_2.pkl"
        cost_path="/home/libe2152/deep-al/usavars/cost_population/${cost_file}"

        r_part=$(echo $cost_file | grep -oP 'r1_\K[0-9]+_r2_[0-9]+')
        cost_part=$(echo $cost_file | grep -oP 'cost_\K[0-9+_vs_0-9]+')

        for budget in 100 200 300 400 500 1000
        do
            for method in greedycost
            do
                for train_seed in 1 42 123 456 789
                do
                    CUDA_VISIBLE_DEVICES=0 python train_al.py \
                        --cfg=../configs/usavars/al/RIDGE_${label}.yaml \
                        --al=${method} \
                        --budget=${budget} \
                        --max_iter=1 \
                        --exp-name=USAVARS_${label}_AL_${method^^}_BUDGET_${budget}_SEED_${train_seed}_IDPATH_${id_file_no_ext}_R${r_part}_COST${cost_part} \
                        --seed=${train_seed} \
                        --id-path=${id_path} \
                        --cost_path=${cost_path}
                done
            done
        done
    done
done
