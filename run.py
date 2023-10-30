import os

n_etimators = [100,150,200,250,280]
max_depth = [5,10,12,15,18]
ml=2
for n in n_etimators:
    for md in max_depth:
        os.system(f"python basic_ml_model.py -n {n} -md {md} -ml {ml}")