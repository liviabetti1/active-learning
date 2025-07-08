#!/bin/bash

# === Configuration ===
LOGFILE="completed_experiments.log"
FAILED_LOG="failed_experiments.log"
EMAIL="liviabetti1@gmail.com"  # <-- replace with your email

# Path to your send_email.py script (adjust if needed)
SEND_EMAIL_PY="send_email.py"

# === Optional: Email on error ===
send_error_email() {
    local exp_name="$1"
    # Compose email subject and body
    local subject="AL Experiment Error: $exp_name"
    local body="Experiment failed: $exp_name
Label: $LABEL_SHORT
Method: $METHOD
Budget: $BUDGET
Seed: $SEED
Initial: $INIT_STR
Line: $LINENO"

    # Call Python send_email.py script
    python "$SEND_EMAIL_PY" "$subject" "$body" "$EMAIL"
}

# === Run one experiment ===
run_experiment() {
    local cmd="$1"
    local exp_name="$2"

    # Skip if already completed
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
for LABEL_SHORT in TC INC; do
    LABEL_LONG=${LABEL_MAP[$LABEL_SHORT]}
    CFG_NAME="RIDGE_${LABEL_SHORT}"
    BASE_DIR="/home/libe2152/deep-al/usavars/${LABEL_LONG}"
    ID_DIRS=("${BASE_DIR}/clustered_initial_sample" "${BASE_DIR}/convenience_initial_sample")
    NLCD_GROUP_PATH="${BASE_DIR}/nlcd_assignments/${LABEL_LONG}_NLCD_cluster_assignments_8.pkl"
    STATE_GROUP_PATH="${BASE_DIR}/region_assignments/state.pkl"
    SIM_MATRIX_PATH="/home/libe2152/deep-al/torchgeo/usavars/${LABEL_LONG}_cosine_similarity_train_test.npz"

    for METHOD in random stratified match_population_proportion poprisk similarity; do

        case $METHOD in
            stratified | match_population_proportion | poprisk)
                GROUP_TYPES=("state" "nlcd")
                ;;
            *)
                GROUP_TYPES=("")
                ;;
        esac

        for GROUP_TYPE in "${GROUP_TYPES[@]}"; do

            if [[ "$GROUP_TYPE" == "nlcd" ]]; then
                GROUP_PATH=$NLCD_GROUP_PATH
            elif [[ "$GROUP_TYPE" == "state" ]]; then
                GROUP_PATH=$STATE_GROUP_PATH
            else
                GROUP_PATH=""
            fi

            for ID_DIR in "${ID_DIRS[@]}"; do
                for ID_FILE in "$ID_DIR"/*.pkl; do

                    INIT_STR=$(basename "$ID_FILE" .pkl)
                    INIT_STR=${INIT_STR#IDS_}
                    INIT_STR=${INIT_STR#sampled_ids_}

                    for BUDGET in $(seq 10 10 100); do #maybe add 200 to 1000
                        for SEED in 1 42 123 456 789; do

                            EXP_NAME="al_${METHOD}_${LABEL_SHORT}_b${BUDGET}_s${SEED}_${INIT_STR}"
                            if [[ "$GROUP_TYPE" != "" ]]; then
                                EXP_NAME="${EXP_NAME}_${GROUP_TYPE}"
                            fi

                            CMD="CUDA_VISIBLE_DEVICES=0 python train_al.py \
--cfg=../configs/usavars/al/${CFG_NAME}.yaml \
--al=${METHOD} \
--budget=${BUDGET} \
--max_iter=1 \
--initial_set_str=${INIT_STR} \
--exp-name=${EXP_NAME} \
--seed=${SEED} \
--id-path=${ID_FILE}"

                            if [[ "$GROUP_TYPE" != "" ]]; then
                                CMD="${CMD} --group_type=${GROUP_TYPE} --group_assignment_path=${GROUP_PATH}"
                            fi

                            if [[ "$METHOD" == "poprisk" || "$METHOD" == "similarity" ]]; then
                                CMD="${CMD} --cost_aware=True --cost_func=uniform"
                            else
                                CMD="${CMD} --cost_aware=False"
                            fi

                            if [[ "$METHOD" == "similarity" ]]; then
                                CMD="${CMD} --similarity_matrix_path=${SIM_MATRIX_PATH}"
                            fi

                            if [[ "$METHOD" == "poprisk" ]]; then
                                CMD="${CMD} --util_lambda=0.5"
                            fi

                            run_experiment "$CMD" "$EXP_NAME"

                        done
                    done
                done
            done
        done
    done
done
