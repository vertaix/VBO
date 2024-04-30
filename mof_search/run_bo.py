import argparse
from utils import *


parser = argparse.ArgumentParser(description="run Bayesian optimization for MOF search")
parser.add_argument("--method", type=str, choices=["random", "BO", "VBO"])
parser.add_argument("--target", type=str, choices=["M_Storage", "M_DBD", "M_safety"])
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

target = args.target
method = args.method
seed = args.seed


train_ind = torch.tensor(train_ind, dtype=torch.long)

train_y = torch.tensor(train_df[target].values)
if target == "M_DBD":
    train_y = torch.log(train_y)
if target == "M_safety":
    train_y = torch.log(train_y + 1)


num_init_queries = 2
num_queries = 10
batch_size = 5
diversity_threshold = 0.9

rel_train_ind = torch.arange(num_init_queries, dtype=torch.long) + seed * num_init_queries
this_train_ind = train_ind[rel_train_ind]
this_train_y = train_y[rel_train_ind]

rel_test_ind = torch.tensor(
    np.delete(
        np.arange(train_y.numel()), rel_train_ind.detach().numpy()
    ),
    dtype=torch.long
)
this_test_ind = train_ind[rel_test_ind]

# run search
for i in range(num_queries // batch_size):
    print(f"iteration {i}: incumbent = {this_train_y.max().item():.4f}")
    
    model, likelihood = fit_gp_model(this_train_ind, this_train_y)
    
    policy = botorch.acquisition.analytic.UpperConfidenceBound(
        model, beta=2
    )
    
    acq_values = []
    with torch.no_grad():
        for cand_ind in this_test_ind:
            acq_values.append(
                policy(torch.atleast_2d(cand_ind)).item()
            )
    acq_values = torch.tensor(acq_values)
    
    if method == "random":
        next_abs_ind = np.random.choice(
            np.arange(len(acq_values)), 
            size=batch_size, 
            replace=False,
        )
        next_rel_ind = rel_test_ind[next_abs_ind]
        next_ind = this_test_ind[next_abs_ind]
    elif method == "BO":
        next_abs_ind = torch.topk(acq_values, k=batch_size).indices
        next_rel_ind = rel_test_ind[next_abs_ind]
        next_ind = this_test_ind[next_abs_ind]
    elif method == "VBO":
        next_abs_ind = [acq_values.argmax().item()]
        next_rel_ind = [rel_test_ind[next_abs_ind].item()]
        next_ind = [train_ind[next_rel_ind].item()]
        for batch_i in range(batch_size - 1):
            # compute Vendi scores when adding each candidate
            vendi_scores = []
            for cand_ind in this_test_ind:
                union_ind = next_ind + [cand_ind.item()]
                vendi_scores.append(
                    calculate_vendi_score(
                        fixed_covariance_matrix[union_ind, :][:, union_ind]
                    )
                )
            vendi_scores = torch.tensor(vendi_scores)
            
            # only keep the top candidates
            cutoff = int(
                (vendi_scores.numel() - len(next_ind)) * diversity_threshold
            )
            
            top_vendi_abs_ind = torch.topk(vendi_scores, k=cutoff).indices
            filtered_acq_values = acq_values[top_vendi_abs_ind]
            
            # add the next candidate
            next_abs_ind.append(
                top_vendi_abs_ind[filtered_acq_values.argmax()].item()
            )
            next_rel_ind = rel_test_ind[next_abs_ind].detach().numpy().tolist()
            next_ind = train_ind[next_rel_ind].detach().numpy().tolist()
    
    rel_train_ind = torch.cat([rel_train_ind, next_rel_ind])
    this_train_ind = train_ind[rel_train_ind]
    this_train_y = train_y[rel_train_ind]
    
    rel_test_ind = torch.tensor(
        np.delete(
            np.arange(train_y.numel()), rel_train_ind.detach().numpy()
        ),
        dtype=torch.long
    )
    this_test_ind = train_ind[rel_test_ind]


if target == "M_Storage":
    folder = "M_Storage"
elif target == "M_DBD":
    folder = "log M_DBD"
elif target == "M_safety":
    folder = "log M_Safety+1"
path = f"../data/simulated_runs/{folder}/{method}"

if not os.path.exists(path):
    os.makedirs(path)

torch.save(this_train_y, os.path.join(path, f"this_train_y_{seed}.pth"))
torch.save(this_train_ind, os.path.join(path, f"this_train_ind_{seed}.pth"))
