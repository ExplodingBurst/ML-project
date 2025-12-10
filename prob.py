import time, math, random
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering

# -------------------------------
# 1) Gridworld + features
# -------------------------------
W, H = 10, 10
START = (0, 0)
GOAL  = (3, 3)
WALLS = {(2,2),(2,3),(2,4),(3,4),(4,4),(4,3),(4,2),(5,5),(5,6),(5,7),(6,7),(7,7)}

ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # down, up, right, left

def in_bounds(x,y): return 0 <= x < W and 0 <= y < H
def blocked(x,y):   return (x,y) in WALLS

def step_state(s, a):
    x,y = s
    dx,dy = a
    nx, ny = x+dx, y+dy
    if not in_bounds(nx,ny) or blocked(nx,ny):
        # optional bump penalty by staying in place
        return (x,y), -0.01, False
    ns = (nx,ny)
    if ns == GOAL:
        return ns, 1.0, True
    return ns, -0.01, False

def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

# All non-wall states
STATES = [(x,y) for x in range(W) for y in range(H) if (x,y) not in WALLS]

def min_wall_dist(s):
    if not WALLS: return 0
    return min(manhattan(s,w) for w in WALLS)

def state_features(s):
    x,y = s
    d_goal = manhattan(s, GOAL) / (W+H-2)       # normalize by max manhattan (~12)
    wall_d = (min_wall_dist(s) / (max(W,H)-1))  # normalize by 6
    return np.array([x/(W-1), y/(H-1), d_goal, wall_d], dtype=float)

# BFS shortest-path distance to goal (ignoring step cost)
def bfs_distance_to_goal():
    dist = {s: math.inf for s in STATES}
    dist[GOAL] = 0
    q = deque([GOAL])
    while q:
        s = q.popleft()
        for a in ACTIONS:
            # reverse neighbors: from neighbor to s
            x,y = s[0]-a[0], s[1]-a[1]
            if in_bounds(x,y) and not blocked(x,y):
                n = (x,y)
                if dist[n] > dist[s] + 1:
                    dist[n] = dist[s] + 1
                    q.append(n)
    return dist

DIST_MAP = bfs_distance_to_goal()

# Build dataset (features, labels) for supervised & unsupervised
X = np.stack([state_features(s) for s in STATES])
y = np.array([DIST_MAP[s] for s in STATES], dtype=float)

# -------------------------------
# 2) Supervised: Linear Regression with NumPy optimizers
# -------------------------------
def train_val_test_split(X, y, test_size=0.2, val_size=0.2, seed=0):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size+val_size, random_state=seed)
    rel_val = val_size / (test_size+val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel_val, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def zscore_train(X_train, X_val, X_test):
    sc = StandardScaler().fit(X_train)
    return sc.transform(X_train), sc.transform(X_val), sc.transform(X_test)

def supervised_run(optimizer="sgd", lr=1e-2, momentum=0.9, beta2=0.999, wd=0.0,
                   epochs=300, eps=0.01, seed=0, grad_clip=None, small_eps=1e-6,
                   val_size=0.2, test_size=0.2):
    rng = np.random.default_rng(seed)
    Xtr, Xva, Xte, ytr, yva, yte = train_val_test_split(X, y, seed=seed, val_size=val_size, test_size=test_size)
    Xtr, Xva, Xte = zscore_train(Xtr, Xva, Xte)
    n, d = Xtr.shape
    # Linear model y_hat = X w + b
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    # States for optimizers
    v_w = np.zeros_like(w)       # momentum / first moment
    v_b = 0.0
    s_w = np.zeros_like(w)       # second moment
    s_b = 0.0
    t = 0
    best_val = float('inf')
    time_to_eps = None
    t0 = time.perf_counter()

    def val_rmse():
        yhat = Xva @ w + b
        return math.sqrt(mean_squared_error(yva, yhat))

    # Adjust conservative defaults per optimizer to avoid early divergence
    if optimizer == "rmsprop" and lr >= 1e-3:
        lr = 1e-4
    if optimizer in ("adamw",) and lr >= 1e-3:
        lr = 1e-4

    for epoch in range(1, epochs+1):
        # full-batch gradient (convex; simple & stable)
        yhat = Xtr @ w + b
        err  = yhat - ytr
        # gradients (full-batch)
        gw = (2.0/len(Xtr)) * (Xtr.T @ err) + 2*wd*w
        gb = (2.0/len(Xtr)) * np.sum(err)

        # Optional gradient clipping (component-wise) to stabilize first steps
        if grad_clip is not None:
            gw = np.clip(gw, -grad_clip, grad_clip)
            gb = float(max(min(gb, grad_clip), -grad_clip))

        t += 1
        if optimizer == "sgd":
            w -= lr * gw
            b -= lr * gb
        elif optimizer == "sgdm":
            v_w = momentum * v_w + gw
            v_b = momentum * v_b + gb
            w  -= lr * v_w
            b  -= lr * v_b
        elif optimizer == "rmsprop":
            alpha = 0.99  # could tune
            s_w = alpha * s_w + (1-alpha) * (gw*gw)
            s_b = alpha * s_b + (1-alpha) * (gb*gb)
            # use small_eps for numerical stability
            w -= (lr / (np.sqrt(s_w) + small_eps)) * gw
            b -= (lr / (math.sqrt(s_b) + small_eps)) * gb
        elif optimizer == "adamw":
            beta1 = 0.9
            # update second moments and first moments
            s_w = beta2 * s_w + (1-beta2) * (gw*gw)
            s_b = beta2 * s_b + (1-beta2) * (gb*gb)
            v_w = beta1 * v_w + (1-beta1) * gw
            v_b = beta1 * v_b + (1-beta1) * gb
            # bias-correct (t must be >0)
            v_w_hat = v_w / (1 - beta1**t)
            v_b_hat = v_b / (1 - beta1**t)
            s_w_hat = s_w / (1 - beta2**t)
            s_b_hat = s_b / (1 - beta2**t)
            # decoupled weight decay (apply multiplicatively as safer alternative)
            if wd:
                w *= (1.0 - lr * wd)
            # update with stability epsilon
            w -= lr * v_w_hat / (np.sqrt(s_w_hat) + small_eps)
            b -= lr * v_b_hat / (math.sqrt(s_b_hat) + small_eps)
        else:
            raise ValueError("optimizer not recognized")

        vloss = val_rmse()
        # detect divergence (nan/inf in parameters)
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            # mark as diverged and break early
            best_val = float('inf')
            time_to_eps = None
            print(f"Diverged on optimizer={optimizer} at epoch={epoch}")
            break
        if vloss < best_val:
            best_val = vloss
        if time_to_eps is None and vloss <= (best_val + eps):
            time_to_eps = time.perf_counter() - t0  # time when we got within eps of best-so-far

    # final test metrics at the end (or you can track best-epoch weights)
    yhat_te = Xte @ w + b
    rmse = math.sqrt(mean_squared_error(yte, yhat_te))
    mae  = float(np.mean(np.abs(yhat_te - yte)))
    r2   = r2_score(yte, yhat_te)
    return {"optimizer": optimizer, "rmse": float(rmse), "mae": mae, "r2": float(r2),
            "time_to_eps_s": time_to_eps, "best_val_seen": float(best_val),
            "target_val_rmse": float(best_val + eps) if best_val != float('inf') else None}

def closed_form_linear(val_size=0.2, test_size=0.2, seed=0, ridge=1e-6):
    Xtr, Xva, Xte, ytr, yva, yte = train_val_test_split(X, y, seed=seed, val_size=val_size, test_size=test_size)
    Xtr, Xva, Xte = zscore_train(Xtr, Xva, Xte)
    Xb = np.concatenate([Xtr, np.ones((len(Xtr), 1))], axis=1)
    A = Xb.T @ Xb + ridge * np.eye(Xb.shape[1])
    wb = np.linalg.solve(A, Xb.T @ ytr)
    w, b = wb[:-1], wb[-1]
    def metrics(Xs, ys):
        yhat = Xs @ w + b
        return (math.sqrt(mean_squared_error(ys, yhat)),
                float(np.mean(np.abs(yhat - ys))),
                r2_score(ys, yhat))
    rmse, mae, r2 = metrics(Xte, yte)
    v_rmse, v_mae, v_r2 = metrics(Xva, yva)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2),
            "val_rmse": float(v_rmse), "val_mae": float(v_mae), "val_r2": float(v_r2)}

# -------------------------------
# 3) Unsupervised: Hierarchical clustering (Agglomerative)
# -------------------------------
def hierarchical_run(linkage="ward", k=3, seed=0):
    # use z-scored features (like supervised)
    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    t0 = time.perf_counter()
    # Ward requires Euclidean and is defined for variance minimization; for others, set affinity='euclidean'
    if linkage == "ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    else:
        # recent sklearn versions no longer accept `affinity` kw; use default Euclidean
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(Xz)
    elapsed = time.perf_counter() - t0
    # Quality
    sil = silhouette_score(Xz, labels, metric="euclidean")
    ch  = calinski_harabasz_score(Xz, labels)
    return {"linkage": linkage, "silhouette": sil, "calinski_harabasz": ch, "fit_time_s": elapsed}

# -------------------------------
# 4) Reinforcement Learning: Q-Learning + Double Q
# -------------------------------
STATE_IDX = {s:i for i,s in enumerate(STATES)}
N_S = len(STATES)
N_A = len(ACTIONS)

def env_reset():
    return START

def env_step(s_idx, a_idx):
    s = STATES[s_idx]
    ns, r, done = step_state(s, ACTIONS[a_idx])
    return STATE_IDX[ns], r, done

def q_learning(episodes=2000, alpha=0.5, gamma=0.95, eps_start=0.3, eps_end=0.05, eps_decay_episodes=1000, alpha_decay=None, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_S, N_A))
    returns = []
    for ep in range(1, episodes+1):
        s = STATE_IDX[env_reset()]
        eps = max(eps_end, eps_start - (eps_start-eps_end)*min(1.0, ep/eps_decay_episodes))
        a = rng.integers(N_A) if rng.random() < eps else int(np.argmax(Q[s]))
        total = 0.0
        for t in range(200):
            ns, r, done = env_step(s, a)
            na = int(np.argmax(Q[ns]))
            td_target = r + gamma * Q[ns, na] * (0.0 if done else 1.0)
            Q[s, a] += alpha * (td_target - Q[s, a])
            total += r
            s = ns
            a = rng.integers(N_A) if rng.random() < eps else int(np.argmax(Q[s]))
            if done: break
        returns.append(total)
        if alpha_decay is not None:
            alpha *= alpha_decay
    return np.array(returns)

def double_q_learning(episodes=2000, alpha=0.5, gamma=0.95, eps_start=0.3, eps_end=0.05, eps_decay_episodes=1000, seed=0):
    rng = np.random.default_rng(seed)
    QA = np.zeros((N_S, N_A))
    QB = np.zeros((N_S, N_A))
    returns = []
    for ep in range(1, episodes+1):
        s = STATE_IDX[env_reset()]
        eps = max(eps_end, eps_start - (eps_start-eps_end)*min(1.0, ep/eps_decay_episodes))
        a = rng.integers(N_A) if rng.random() < eps else int(np.argmax(QA[s] + QB[s]))
        total = 0.0
        for t in range(200):
            ns, r, done = env_step(s, a)
            if rng.random() < 0.5:
                # update QA using action from QA, value from QB
                na = int(np.argmax(QA[ns]))
                td_target = r + gamma * QB[ns, na] * (0.0 if done else 1.0)
                QA[s, a] += alpha * (td_target - QA[s, a])
            else:
                # update QB using action from QB, value from QA
                na = int(np.argmax(QB[ns]))
                td_target = r + gamma * QA[ns, na] * (0.0 if done else 1.0)
                QB[s, a] += alpha * (td_target - QB[s, a])
            total += r
            s = ns
            # pick next action from combined estimates
            a = rng.integers(N_A) if rng.random() < eps else int(np.argmax(QA[s] + QB[s]))
            if done: break
        returns.append(total)
    return np.array(returns)

# -------------------------------
# 5) Example runs (pilot)
# -------------------------------
if __name__ == "__main__":
    VAL_SIZE = 0.1
    TEST_SIZE = 0.2

    # Closed-form ridge baseline for sanity
    ols = closed_form_linear(val_size=VAL_SIZE, test_size=TEST_SIZE, seed=0, ridge=1e-6)
    print("Supervised (closed-form ridge):", ols)

    # Supervised pilot: compare optimizers
    for opt in ["sgd", "sgdm", "rmsprop", "adamw"]:
        # conservative per-optimizer defaults
        if opt == "rmsprop":
            res = supervised_run(optimizer=opt, lr=5e-5, wd=0.0, epochs=400, eps=0.01, seed=0,
                                 grad_clip=0.5, val_size=VAL_SIZE, test_size=TEST_SIZE)
        elif opt == "adamw":
            res = supervised_run(optimizer=opt, lr=5e-5, wd=0.0, epochs=400, eps=0.01, seed=0,
                                 grad_clip=0.5, val_size=VAL_SIZE, test_size=TEST_SIZE)
        else:
            res = supervised_run(optimizer=opt, lr=1e-2, wd=1e-4 if opt in ["sgd","sgdm"] else 0.0,
                                 epochs=300, eps=0.01, seed=0, val_size=VAL_SIZE, test_size=TEST_SIZE)
        print("Supervised:", res)

    # Hierarchical: compare linkages
    for lk in ["ward","average","complete"]:
        res = hierarchical_run(linkage=lk, k=3, seed=0)
        print("Hierarchical:", res)

    # RL: Q-learning variants
    t0 = time.perf_counter()
    q_const = q_learning(episodes=1500, alpha=0.5, gamma=0.95, seed=0)
    t_const = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_decay = q_learning(episodes=1500, alpha=0.5, gamma=0.95, alpha_decay=0.999, seed=0)
    t_decay = time.perf_counter() - t0

    t0 = time.perf_counter()
    dq      = double_q_learning(episodes=1500, alpha=0.5, gamma=0.95, seed=0)
    t_dq = time.perf_counter() - t0

    def moving_avg(x, w=100):
        x = np.array(x, dtype=float)
        if len(x)<w: return np.array([])
        return np.convolve(x, np.ones(w)/w, mode='valid')

    ma_const = moving_avg(q_const)
    ma_decay = moving_avg(q_decay)
    ma_dq    = moving_avg(dq)

    def episodes_to_target(ma, target, window):
        idx = np.where(ma >= target)[0]
        return int(idx[0] + window - 1) if len(idx)>0 else None

    # rough optimal return estimate: shortest path length steps with -0.01 each + 1 on goal
    opt_steps = DIST_MAP[START]
    opt_return = 1.0 - 0.01*opt_steps
    target = 0.9 * opt_return

    rl_window = 100
    rl_summary = {
        "const": {"episodes_to_target": episodes_to_target(ma_const, target, rl_window),
                  "wall_time_s": t_const},
        "decay": {"episodes_to_target": episodes_to_target(ma_decay, target, rl_window),
                  "wall_time_s": t_decay},
        "doubleQ": {"episodes_to_target": episodes_to_target(ma_dq, target, rl_window),
                    "wall_time_s": t_dq},
    }
    print("RL episodes-to-target:", rl_summary)
