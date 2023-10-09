import os

def best(all_folder):
    best = {"nifty": {"d": "", "loss": 500}, "axis": {"d": "", "loss": 500}, "tata": {"d": "", "loss": 500}}
    for item in os.listdir(all_folder):
        item_path = os.path.join(all_folder, item)
        if not os.path.isfile(item_path + "/test_loss.txt"):
            continue
        with open(item_path + "/test_loss.txt") as f:
            loss = float(f.read().split("\n")[0].split(" = ")[1])
            if "axis" in item_path:
                if loss < best["axis"]["loss"]:
                    best["axis"]["loss"] = loss
                    best["axis"]["d"] = item_path
            elif "nifty" in item_path:
                if loss < best["nifty"]["loss"]:
                    best["nifty"]["loss"] = loss
                    best["nifty"]["d"] = item_path
            else:
                if loss < best["tata"]["loss"]:
                    best["tata"]["loss"] = loss
                    best["tata"]["d"] = item_path
    
    print(best)


best("all")