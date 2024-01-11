import json

def log_metrics(metrics, ckpt_dir, h, n_params = None):
    with open(ckpt_dir + "/metrics.txt", 'w+') as f:
        f.write("hyperparameters:\n")
        json.dump(h, f)
        if (n_params != None):
            f.write("\n\n")
            f.write("n_params="+str(n_params))
        f.write("\n\n")
        f.write("metrics:")
        json.dump(metrics[0], f)