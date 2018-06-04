##############################################
# desc : root
# @author : qiangz2012@yeah.net
##############################################

from load_data import *
from dnn_mlp import *

def run_mlp_model(mlp_params, train_reader):
    # train MLP model
    mlp = Dnn_Mlp(**mlp_params)
    mlp.train(train_reader)
    # predict

    

if __name__ == "__main__":
    # load data
    train_reader = DataLoader("../data/train.csv")
    train_reader._load_data()
    #train_reader._test_data()

    # run mlp model
    mlp_parms = {
        "epoch":10,
	"batch_size":16,
	"lr":0.0005,
	"optimizer":"adam",
        "l2_reg":0.01,
        "hidden_units":64,
        "output_units":32,
        "feat_size":7
    }
    run_mlp_model(mlp_parms,train_reader)
    
