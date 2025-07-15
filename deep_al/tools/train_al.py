import os
import sys
from datetime import datetime
import argparse
import numpy as np
import dill
import json

import torch
from copy import deepcopy

# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


logger = lu.get_logger(__name__)

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

delta_avg_lst = []
delta_std_lst = []


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str)
    parser.add_argument('--cost_aware', help='Whether or not a cost aware method is used', required=False, type=str2bool)
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--id-path', help='IDs for initial set', default=None, type=str)
    parser.add_argument('--seed', help='Random seed', required=True, default=1, type=int)

    parser.add_argument('--initial_set_str', default=None, type=str)
    parser.add_argument('--cost_func', default=None, type=str)
    parser.add_argument('--cost_name', default=None, type=str)
    parser.add_argument('--cost_array_path', default=None, type=str)

    parser.add_argument('--group_type', default=None, type=str)
    parser.add_argument('--group_assignment_path', default=None, type=str)

    parser.add_argument('--unit_type', default=None, type=str)
    parser.add_argument('--points_per_unit', default=None, type=int)
    parser.add_argument('--unit_assignment_path', default=None, type=str)
    parser.add_argument('--unit_cost_path', default=None, type=str)

    parser.add_argument('--util_lambda', default=None, type=float, help='Value of lambda in pop risk utility function')
    parser.add_argument('--similarity_matrix_path', default=None, type=str)
    parser.add_argument('--distance_matrix_path', default=None, type=str)

    return parser


def main(cfg):
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available() and cfg.MODEL.TYPE != 'ridge'

    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)

    label_mapping = {
        "POP": "population",
        "TC": "treecover",
        "INC": "income"
        # Add more if needed
    }

    dataset_parts = cfg.DATASET.NAME.split('_')
    if len(dataset_parts) == 2:
        dataset_root = dataset_parts[0]  # e.g., "USAVARS"
        label_key = dataset_parts[1]     # e.g., "POP"
        label_name = label_mapping.get(label_key, label_key.lower())
    else:
        dataset_root = cfg.DATASET.NAME
        label_name = "unknown"

    dataset_out_dir = os.path.join(cfg.OUT_DIR, dataset_root, label_name)

    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)

    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        if cfg.ACTIVE_LEARNING.COST_AWARE:
            if cfg.GROUPS.GROUP_TYPE is not None:
                exp_dir = f'{cfg.INITIAL_SET.STR}/cost_aware/{cfg.COST.NAME}/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/{cfg.GROUPS.GROUP_TYPE}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}/seed_{cfg.RNG_SEED}'
            else:
                exp_dir = f'{cfg.INITIAL_SET.STR}/cost_aware/{cfg.COST.NAME}/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}/seed_{cfg.RNG_SEED}'
        else:
            if cfg.GROUPS.GROUP_TYPE is not None:
                exp_dir = f'{cfg.INITIAL_SET.STR}/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/{cfg.GROUPS.GROUP_TYPE}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}/seed_{cfg.RNG_SEED}'
            else:
                exp_dir = f'{cfg.INITIAL_SET.STR}/{cfg.ACTIVE_LEARNING.SAMPLING_FN}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}/seed_{cfg.RNG_SEED}'

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir
    cfg.EXP_ROOT = os.path.dirname(exp_dir)
    cfg.INITIAL_SET_DIR = os.path.join(dataset_out_dir, cfg.INITIAL_SET.STR)

    dump_cfg(cfg)

    lu.setup_logging(cfg)

    print("\n======== PREPARING DATA AND MODEL ========\n")

    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    cfg.ACTIVE_LEARNING.INIT_L_NUM = args.initial_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    if cfg.LSET_IDS:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids(cfg.LSET_IDS, data=train_data, save_dir=cfg.EXP_DIR)
    else:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_num=cfg.ACTIVE_LEARNING.INIT_L_NUM, \
            val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)

    model = model_builder.build_model(cfg).cuda() if use_cuda else model_builder.build_model(cfg)

    al_iter = 0
    episode = 0
    if len(lSet) == 0:
        print('Labeled Set is Empty - Sampling an Initial Pool')
        al_obj = ActiveLearning(data_obj, cfg)

        if cfg.ACTIVE_LEARNING.COST_AWARE:
            if cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
                activeSet, new_uSet, total_cost = al_obj.sample_from_uSet(model, lSet, uSet)
            else:
                activeSet, new_uSet, total_cost, probs, relevant_indices = al_obj.sample_from_uSet(model, lSet, uSet)
        else:
            activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
        print(f'Initial Pool is {activeSet}')
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet
        episode = 1
        al_iter += 1

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    #valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    # Initialize the model.  
    model = model_builder.build_model(cfg)
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    if cfg.MODEL.TYPE != 'ridge':
        optimizer = optim.construct_optimizer(cfg, model)
        opt_init_state = deepcopy(optimizer.state_dict())
        model_init_state = deepcopy(model.state_dict().copy())

        print("optimizer: {}\n".format(optimizer))
        logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    for cur_episode in range(episode, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        # Train model
        print("======== TRAINING ========")
        logger.info("======== TRAINING ========")

        if cfg.MODEL.TYPE != 'ridge':
            raise ValueError
        else:
            episode_0_summary = os.path.join(cfg.INITIAL_SET_DIR, "episode_0/summary.json")
            r2 = None
            if os.path.exists(episode_0_summary) and cur_episode == 0:
                print("Loading previous initial set results...")
                try:
                    with open(episode_0_summary, "rb") as f:
                        raw_r2 = json.load(f).get('test_r2', None)
                        if isinstance(raw_r2, str):
                            raw_r2 = raw_r2.strip().rstrip('.').strip()
                        r2 = float(raw_r2)
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    print(f"Warning: Failed to load or parse r2 from {episode_0_summary}: {e}")
                    r2 = None
                    
            if r2 is None:
                labeled_indices = lSet.tolist()
                X_train = train_data[labeled_indices][0]
                y_train = train_data[labeled_indices][1]

                test_indices = np.arange(len(test_data))
                X_test = test_data[test_indices][0]
                y_test = test_data[test_indices][1]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),     # Step 1: Standardize features
                    ('ridgecv', RidgeCV(alphas=np.logspace(-5,5,10), scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42)))  # Step 2: RidgeCV with 5-fold CV
                ])

                model = pipeline
                model.fit(X_train, y_train)

                best_alpha = model.named_steps['ridgecv'].alpha_
                print(f"Best alpha: {best_alpha}")

                r2 = model.score(X_test, y_test)

                if cur_episode == 0:
                    os.makedirs(os.path.dirname(episode_0_summary), exist_ok=True)
                    with open(episode_0_summary, "w") as f:
                        json.dump({'test_r2': r2}, f)
                    print(f"Saved r2={r2:.4f} to {episode_0_summary}")

            print("Test Accuracy: {}.\n".format(round(r2, 4)))
            logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, r2))
            plot_episode_yvalues.append(r2)

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        if al_iter >= cfg.ACTIVE_LEARNING.MAX_ITER:
            break
        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")
        al_obj = ActiveLearning(data_obj, cfg)

        if cfg.MODEL.TYPE != 'ridge':
            clf_model = model_builder.build_model(cfg)
            clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
        else:
            clf_model = model
        
        if cfg.ACTIVE_LEARNING.COST_AWARE:
            if cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
                activeSet, new_uSet, total_cost = al_obj.sample_from_uSet(model, lSet, uSet)
            else:
                activeSet, new_uSet, total_cost, probs, relevant_indices = al_obj.sample_from_uSet(model, lSet, uSet)
                latlons = [train_data[idx][2] for idx in relevant_indices]
                ids = [train_data[idx][3] for idx in relevant_indices]
                pkl_file = os.path.join(cfg.EXP_ROOT, "probabilities.pkl")
                with open(pkl_file, "wb") as f:
                    dill.dump({"ids": ids, "latlons": latlons, "probs":probs}, f)
        else:
            activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet
        al_iter += 1

        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        #valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))

        if cfg.ACTIVE_LEARNING.COST_AWARE:
            print("Total Cost of New Labeled Set: {}".format(total_cost))
            logger.info("Total Cost of New Labeled Set: {}".format(total_cost))

        print("================================\n\n")
        logger.info("================================\n\n")

        print('Current accuracy values: ', plot_episode_yvalues)

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            if cfg.MODEL.TYPE != 'ridge':
                # start model from scratch
                print('Starting model from scratch - ignoring existing weights.')
                model = model_builder.build_model(cfg)
                # Construct the optimizer
                optimizer = optim.construct_optimizer(cfg, model)
                print(model.load_state_dict(model_init_state))
                print(optimizer.load_state_dict(opt_init_state))

                os.remove(checkpoint_file)

if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.COST_AWARE = args.cost_aware
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.UTIL_LAMBDA = args.util_lambda
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.INITIAL_SET.STR = args.initial_set_str if args.initial_set_str is not None else "empty_intial_set"
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    cfg.RNG_SEED = args.seed
    cfg.ACTIVE_LEARNING.MAX_ITER = args.max_iter
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic

    cfg.COST.FN = args.cost_func
    cfg.COST.NAME = args.cost_name if args.cost_name is not None else cfg.COST.FN

    cfg.ID_PATH = args.id_path
    if cfg.ID_PATH is not None:
        with open(cfg.ID_PATH, "rb") as f:
            loaded_ids = dill.load(f)

        cfg.LSET_IDS = loaded_ids if isinstance(loaded_ids, list) else loaded_ids.tolist()
        cfg.INIT_L_NUM = len(loaded_ids)
    else:
        cfg.LSET_IDS = []
        cfg.INIT_L_NUM = 0

    group_assignment_path = args.group_assignment_path
    unit_assignment_path = args.unit_assignment_path

    if group_assignment_path is not None:
        group_type = args.group_type
        cfg.GROUPS.GROUP_TYPE = group_type

        with open(group_assignment_path, "rb") as f:
            loaded_group_assignments = dill.load(f)['assignments']

        cfg.GROUPS.GROUP_ASSIGNMENT = loaded_group_assignments.tolist() if not isinstance(loaded_group_assignments, list) else loaded_group_assignments

    if unit_assignment_path is not None:
        unit_type = args.unit_type
        cfg.UNITS.UNIT_TYPE = unit_type

        with open(unit_assignment_path, "rb") as f:
            loaded_unit_assignments = dill.load(f)['assignments']

        cfg.UNITS.UNIT_ASSIGNMENT = loaded_unit_assignments.tolist() if not isinstance(loaded_unit_assignments, list) else loaded_unit_assignments
        cfg.UNITS.POINTS_PER_UNIT = args.points_per_unit if args.points_per_unit is not None else None

        if args.unit_cost_path is not None:
            with open(args.unit_cost_path, "rb") as f:
                cost_dict = dill.load(f)

            cfg.COST.UNIT_COST = [cost_dict] # just to store in the config file

    if args.cost_array_path is not None:
        with open(args.cost_array_path, "rb") as f:
            cost_array = dill.load(f)['costs']

        cfg.COST.ARRAY = cost_array.tolist() if not isinstance(cost_array, list) else cost_array

    cfg.ACTIVE_LEARNING.SIMILARITY_MATRIX_PATH = args.similarity_matrix_path

    cfg.ACTIVE_LEARNING.DISTANCE_MATRIX_PATH = args.distance_matrix_path

    main(cfg)
