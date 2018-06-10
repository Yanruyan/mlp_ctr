##############################################
# desc : root
# @author : qiangz2012@yeah.net
##############################################

from load_data import *
from dnn_mlp import *

def run_mlp_model(mlp_params, X_reader, Y_reader):
    # train MLP model
    mlp = Dnn_Mlp(**mlp_params)
    #mlp.train(train_reader)
    # predict

    

if __name__ == "__main__":
    # load data
    X_reader = DataLoader("../data/train.csv","X")
    X_reader._load_data()
    #X_reader._test_data() 

    Y_reader = DataLoader("../data/train.csv","Y")
    Y_reader._load_data()
    #Y_reader._test_data()

    # run mlp model
    mlp_parms = {
        "epoch":10,
	"batch_size":16,
	"lr":0.0005,
	"optimizer_type":"adam",
        "l2_reg":0.01,
        "hidden_units":64,
        "output_units":32,
        "feat_size":7,
	"activation":"relu",
	"loss_type":"logloss"
    }
    run_mlp_model(mlp_parms,X_reader,Y_reader)
    



