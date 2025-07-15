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

unset GROUPS
GROUPS=("nlcd" "state")

# === Main experiment loop ===
# === Main experiment loop ===
for LABEL_SHORT in POP TC; do
    LABEL_LONG=${LABEL_MAP[$LABEL_SHORT]}
    CFG_NAME="RIDGE_${LABEL_SHORT}"
    BASE_DIR="/home/libe2152/deep-al/usavars/${LABEL_LONG}"
    GROUP_PATH_NLCD="${BASE_DIR}/nlcd_assignments/${LABEL_LONG}_NLCD_cluster_assignments_8.pkl"
    GROUP_PATH_STATE="${BASE_DIR}/region_assignments/state.pkl"

    SEEDS=(1 42 123 456 789)
    GROUPS=("nlcd" "state")

    for COST_AWARE in True False; do
        INIT_STR="empty_initial_set"
        COST_NAME="uniform"
        COST_ARGS=""
        if [[ "$COST_AWARE" == "True" ]]; then
            COST_ARGS="--cost_func=uniform --cost_name=uniform"
            METHODS=("poprisk")
        else
            METHODS=("random" "stratified" "match_population_proportion")
        fi

        for METHOD in "${METHODS[@]}"; do
            for GROUP_TYPE in "${GROUPS[@]}"; do

                if [[ "$GROUP_TYPE" == "nlcd" ]]; then
                    GROUP_PATH="$GROUP_PATH_NLCD"
                elif [[ "$GROUP_TYPE" == "state" ]]; then
                    GROUP_PATH="$GROUP_PATH_STATE"
                else
                    echo "❌ Unknown group_type: $GROUP_TYPE"
                    exit 1
                fi

                if [[ ! -f "$GROUP_PATH" ]]; then
                    echo "❌ Missing group assignment file: $GROUP_PATH"
                    continue
                fi

                for BUDGET in $(seq 10 10 200); do
                    for SEED in "${SEEDS[@]}"; do
                        EXP_NAME="al_${METHOD}_${LABEL_SHORT}_${COST_NAME}_b${BUDGET}_s${SEED}_${INIT_STR}_${GROUP_TYPE}"
                        if [[ "$COST_AWARE" == "False" ]]; then
                            EXP_NAME="al_${METHOD}_${LABEL_SHORT}_b${BUDGET}_s${SEED}_${INIT_STR}_${GROUP_TYPE}"
                        fi

                        CMD="CUDA_VISIBLE_DEVICES=0 python train_al.py \
--cfg=../configs/usavars/al/${CFG_NAME}.yaml \
--al=${METHOD} \
--budget=${BUDGET} \
--max_iter=1 \
--initial_set_str=${INIT_STR} \
--exp-name=${EXP_NAME} \
--seed=${SEED} \
--group_type=${GROUP_TYPE} \
--group_assignment_path=${GROUP_PATH} \
--cost_aware=${COST_AWARE} \
${COST_ARGS} \
--util_lambda=0.5"

                        run_experiment "$CMD" "$EXP_NAME"
                    done
                done
            done
        done
    done
done
