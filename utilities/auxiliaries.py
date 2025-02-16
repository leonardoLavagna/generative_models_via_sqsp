import numpy as np

def compute_loss(y_hat, y, loss_type="mmd"):
    y_hat = np.array(y_hat)
    y = np.array(y)
    min_len = min(len(y_hat), len(y))
    y_hat = y_hat[:min_len]
    y = y[:min_len]
    if loss_type == "l1":  
        return np.sum(np.abs(y - y_hat))
    elif loss_type == "l2":  
        return np.sqrt(np.sum((y - y_hat) ** 2))
    elif loss_type == "mmd":  
        squared_diff = np.sum((y - y_hat) ** 2)
        euclidean_dist = np.sqrt(squared_diff)
        return euclidean_dist + 1e-6 * np.sum(np.abs(y - y_hat))
    else:
        raise ValueError("Unsupported loss. Use 'l1', 'l2' o 'mmd'.")


  def generate_parameters(n, k=2):
    return list(np.random.uniform(low=0, high=k * np.pi, size=n))


def generate_parameters(n, k=2):
    return list(np.random.uniform(low=0, high=k * np.pi, size=n))


def callback_fn(current_params):
    current_loss = compute_loss(exp_distribution, p_target, loss_type="mmd")
    loss_history.append(current_loss)


def compute_loss_partial(opt_thetas, full_thetas, opt_indices, p_target):
    new_thetas = full_thetas.copy()
    for i, idx in enumerate(opt_indices):
        new_thetas[idx] = opt_thetas[i]
    qc_partial = state_expansion(m, list(new_thetas))
    t_qc_partial = transpile(qc_partial, backend=backend)
    job_partial = backend.run(t_qc_partial, shots=shots)
    counts_partial = job_partial.result().get_counts(qc_partial)
    exp_distribution = np.array([counts_partial.get(state, 0) for state in all_states], dtype=float)
    if exp_distribution.sum() > 0:
        exp_distribution /= exp_distribution.sum()
    return compute_loss(exp_distribution, p_target, loss_type="mmd")
