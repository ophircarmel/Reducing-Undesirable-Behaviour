# if we want to use Snake grammar, turn 'Snake' into - True
SNAKE = False
NO_FEATS = False
PROPERTY = "Properties/PROPERTY 0/"

if SNAKE:
    PERF_DICT = PROPERTY + "perf_dict_snake.pkl"
    if NO_FEATS:
        X_PKL = PROPERTY + "X_snake_no_features.pkl"
        Y_PKL = PROPERTY + "Y_snake_no_features.pkl"
        UNDES_FILE = PROPERTY + 'undes_no_features.txt'
        DES_FILE = PROPERTY + 'des_no_features.txt'
        TREE_PTH = PROPERTY + 'snake_tree_no_features.pkl'
    else:
        X_PKL = PROPERTY + "X_snake.pkl"
        Y_PKL = PROPERTY + "Y_snake.pkl"
        UNDES_FILE = PROPERTY + 'undes_snake.txt'
        DES_FILE = PROPERTY + 'des_snake.txt'
        TREE_PTH = PROPERTY + 'snake_tree.pkl'

    TREE_FOLDER = "Trees_snake/"
else:
    PERF_DICT = PROPERTY + "perf_dict.pkl"
    if NO_FEATS:
        X_PKL = PROPERTY + "X_no_features.pkl"
        Y_PKL = PROPERTY + "Y_no_features.pkl"
        UNDES_FILE = PROPERTY + 'undes_no_features.txt'
        DES_FILE = PROPERTY + 'des_no_features.txt'
        TREE_PTH = PROPERTY + 'tree_no_features.pkl'
    else:
        X_PKL = PROPERTY + "X_new.pkl"
        Y_PKL = PROPERTY + "Y_new.pkl"
        UNDES_FILE = PROPERTY + 'des.txt'
        DES_FILE = PROPERTY + 'undes.txt'
        TREE_PTH = PROPERTY + 'finalized_tree.pkl'
    TREE_FOLDER = "Trees/"