import jax.numpy as jnp
import jax
import jax.lax as lax


def get_solver(name, f, z_init):
    if name == 'fwd_solver':
        return fwd_solver(f, z_init)
    elif name == 'fwd_solver_jitable':
        return fwd_solver(f, z_init)
    elif name == 'newton_solver':
        return newton_solver(f, z_init)
    elif name == 'newton_solver_jitable':
        return newton_solver_jitable(f, z_init)
    elif name == 'anderson_solver':
        return anderson_solver(f, z_init)
    elif name == 'anderson_solver_F':
        return anderson_solver_F(f, z_init)
    elif name == 'anderson_solver_jitable':
        return anderson_solver_jitable(f, z_init)
    elif name == 'anderson_solver_F_jitable':
        return anderson_solver_F_jitable(f, z_init)

    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)


def fwd_solver(f, z_init):
    z_prev, z = z_init, f(z_init)
    while jnp.linalg.norm(z_prev - z) > 1e-5:
        z_prev, z = z, f(z)
    return z

def fwd_solver_jitable(f, z_init):
    i = 0
    def cond_fun(carry):
        z_prev, z, k = carry
        return (jnp.linalg.norm(z_prev - z) > 1e-5) & (k<50)

    def body_fun(carry):
        _, z, k = carry
        return z, f(z), k+1

    init_carry = (z_init, f(z_init), i)
    _, z_star, _ = lax.while_loop(cond_fun, body_fun, init_carry)
    return z_star

def newton_solver_jitable(f, z_init):
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return fwd_solver_jitable(g, z_init)

def newton_solver(f, z_init):
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return fwd_solver(g, z_init)


def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:n] - X[:n]
        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
        H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))], [jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
        alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]
        xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
        if res < tol:
            break
    return xk


def anderson_solver_jitable(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    def step(n, k, X, F):
        G = F[:n] - X[:n]
        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
        H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],[ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
        alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1: ]
        xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        return X, F
    # unroll the first m steps
    for k in range(2, m):
        X, F = step(k, k, X, F)
        res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
        if res < tol or k + 1 >= max_iter:
            return X[k], k
    # run the remaining steps in a lax.while_loop
    def body_fun(carry):
        k, X, F = carry
        X, F = step(m, k, X, F)
        return k + 1, X, F

    def cond_fun(carry):
        k, X, F = carry
        kmod = (k - 1) % m
        res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
        return (k < max_iter) & (res >= tol)

    k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
    return X[(k - 1) % m]


def anderson_solver_F(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    # vmapped version of anderson solver, that allows for batches in the last dimension
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:n] - X[:n]
        GTG = jax.vmap(lambda G: jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2),in_axes=-1, out_axes=-1)(G)
        H = jax.vmap(lambda GTG: jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))], [jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1), in_axes=-1, out_axes=-1)(GTG)
        alpha = jax.vmap(lambda H: jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:], in_axes=-1, out_axes=-1)(H)
        xk = jax.vmap(lambda F, X, alpha: beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n]), in_axes=(-1,-1,-1), out_axes=-1)(F,X,alpha)
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        res = jnp.linalg.norm(F[k % m] - X[k % m], axis=0) / (1e-5 + jnp.linalg.norm(F[k % m], axis=0))
        if jnp.min(res) < tol:
            break
    return xk


# 2nd version of jitable anderson solver
def anderson_solver_F_jitable(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    def step(n, k, X, F):
        G = F[:n] - X[:n]
        GTG = jax.vmap(lambda G: jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2), in_axes=-1, out_axes=-1)(G)
        H = jax.vmap(
            lambda GTG: jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))], [jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(
                n + 1),
            in_axes=-1, out_axes=-1)(GTG)
        alpha = jax.vmap(lambda H: jnp.linalg.solve(H, jnp.zeros(n + 1).at[0].set(1))[1:], in_axes=-1, out_axes=-1)(H)
        xk = jax.vmap(lambda F, X, alpha: beta * jnp.dot(alpha, F[:n]) + (1 - beta) * jnp.dot(alpha, X[:n]),
                      in_axes=(-1, -1, -1), out_axes=-1)(F, X, alpha)
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        return X, F
    # unroll the first m steps
    for k in range(2, m):
        X, F = step(m, k, X, F)
        res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
#        if res < tol or k + 1 >= max_iter:
#            return X[k], k
    # run the remaining steps in a lax.while_loop
    def body_fun(carry):
        k, X, F = carry
        X, F = step(m, k, X, F)
        return k + 1, X, F

    def cond_fun(carry):
        k, X, F = carry
        kmod = (k - 1) % m
        res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
        return (k < max_iter) & (res >= tol)

    k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
    return X[(k - 1) % m]



def anderson_solver_F_jitable_(f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    def step(n, k, X, F):
        G = F[:n] - X[:n]
        GTG = jax.vmap(lambda G: jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2), in_axes=-1, out_axes=-1)(G)
        H = jax.vmap(
            lambda GTG: jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))], [jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(
                n + 1),
            in_axes=-1, out_axes=-1)(GTG)
        alpha = jax.vmap(lambda H: jnp.linalg.solve(H, jnp.zeros(n + 1).at[0].set(1))[1:], in_axes=-1, out_axes=-1)(H)
        xk = jax.vmap(lambda F, X, alpha: beta * jnp.dot(alpha, F[:n]) + (1 - beta) * jnp.dot(alpha, X[:n]),
                      in_axes=(-1, -1, -1), out_axes=-1)(F, X, alpha)
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        return X, F
    # unroll the first m steps
#    for k in range(2, m):
#        X, F = step(m, k, X, F)
#        res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
#        if res < tol or k + 1 >= max_iter:
#            return X[k], k
    # run the remaining steps in a lax.while_loop
    def body_fun(carry):
        k, X, F = carry
        X, F = step(m, k, X, F)
        return k + 1, X, F

    def cond_fun(carry):
        k, X, F = carry
        kmod = (k - 1) % m
        res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
        return (k < max_iter) & (res >= tol)

    k=2
    k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
    return X[(k - 1) % m]