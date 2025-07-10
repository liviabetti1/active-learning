#!/bin/bash

# === Configuration ===
LOGFILE="completed_experiments.log"
FAILED_LOG="failed_experiments.log"
EMAIL="liviabetti1@gmail.com"
SEND_EMAIL_PY="send_email.py"

send_error_email() {
    local exp_name="$1"
    local subject="AL Experiment Error: $exp_name"
    local body="Experiment failed: $exp_name\nLabel: $LABEL_SHORT\nMethod: $METHOD\nBudget: $BUDGET\nSeed: $SEED\nInitial: $INIT_STR\nLine: $LINENO"
    python "$SEND_EMAIL_PY" "$subject" "$body" "$EMAIL"
}

run_experiment() {
    local cmd="$1"
    local exp_name="$2"

    if grep -qx "$exp_name" "$LOGFILE" 2>/dev/null; then
        echo "✅ Skipping completed: $exp_name"
        return
    fi

    echo "▶️ Running: $exp_name"
    eval "$cmd"
    local status=$?

    if [[ $status -ne 0 ]]; then
        echo "❌ Error in experiment: $exp_name"
        echo "$exp_name" >> "$FAILED_LOG"
        send_error_email "$exp_name"
        echo "⚠️ Paused due to error. Press Enter to continue or Ctrl+C to abort."
        read
    else
        echo "$exp_name" >> "$LOGFILE"
    fi
}

# === Label mapping ===
declare -A LABEL_MAP=( ["POP"]="population" ["INC"]="income" ["TC"]="treecover" )

# === Main experiment loop ===
for LABEL_SHORT in POP TC; do
    LABEL_LONG=${LABEL_MAP[$LABEL_SHORT]}
    CFG_NAME="RIDGE_${LABEL_SHORT}"
    BASE_DIR="/home/libe2152/deep-al/usavars/${LABEL_LONG}"
    ID_DIRS=("${BASE_DIR}/clustered_initial_sample" "${BASE_DIR}/convenience_initial_sample")
    GROUP_PATH="${BASE_DIR}/nlcd_assignments/${LABEL_LONG}_NLCD_cluster_assignments_8.pkl"
    UNIT_ASSIGNMENT_PATH="${BASE_DIR}/region_assignments/counties.pkl"

    # Seeds to iterate over
    SEEDS=(1 42 123 456 789)

    for ID_DIR in "${ID_DIRS[@]}"; do
        for ID_FILE in "$ID_DIR"/*.pkl; do

            INIT_STR=$(basename "$ID_FILE" .pkl)
            INIT_STR=${INIT_STR#IDS_}
            INIT_STR=${INIT_STR#sampled_ids_}

            points_per_cluster=$(echo "$INIT_STR" | grep -oP '(?<=_)\d+(?=_points_per_cluster)')
            if [[ -z "$points_per_cluster" ]]; then
                points_per_cluster=""
            fi

            declare -A COST_CONFIGS=(
                ["pointwise_by_array_cluster_based"]="--cost_func=pointwise_by_array --cost_name=pointwise_by_array_cluster_based --unit_assignment_path=${UNIT_ASSIGNMENT_PATH} --unit_cost_path=${BASE_DIR}/cost/county_costs_${points_per_cluster}_points_per_cluster.pkl --points_per_unit=${points_per_cluster}"
                #["unit_aware_pointwise_cost"]="--cost_func=unit_aware_pointwise_cost --unit_assignment_path=${UNIT_ASSIGNMENT_PATH}"
                ["pointwise_by_array_distance_based"]="--cost_func=pointwise_by_array --cost_name=pointwise_by_array_distance_based --cost_array_path=${BASE_DIR}/cost/distance_based_costs_top50_urban.pkl"
            )

            # Decide which cost configs to use based on the ID_DIR
            case "$ID_DIR" in
                *clustered_initial_sample*)
                    COST_KEYS=("pointwise_by_array_cluster_based" "unit_aware_pointwise_cost")
                    ;;
                *convenience_initial_sample*)
                    COST_KEYS=("pointwise_by_array_distance_based")
                    ;;
                *)
                    COST_KEYS=()
                    ;;
            esac

            for COST_NAME in "${COST_KEYS[@]}"; do
                COST_ARGS=${COST_CONFIGS[$COST_NAME]}

                for BUDGET in $(seq 10 10 200); do
                    for SEED in "${SEEDS[@]}"; do

                        EXP_NAME="al_poprisk_${LABEL_SHORT}_${COST_NAME}_b${BUDGET}_s${SEED}_${INIT_STR}_nlcd"

                        CMD="CUDA_VISIBLE_DEVICES=0 python train_al.py \
--cfg=../configs/usavars/al/${CFG_NAME}.yaml \
--al=poprisk \
--budget=${BUDGET} \
--max_iter=1 \
--initial_set_str=${INIT_STR} \
--exp-name=${EXP_NAME} \
--seed=${SEED} \
--id-path=${ID_FILE} \
--group_type=nlcd \
--group_assignment_path=${GROUP_PATH} \
--cost_aware=True \
${COST_ARGS} \
--util_lambda=0.5"

                        run_experiment "$CMD" "$EXP_NAME"

                    done
                done
            done

        done
    done
done