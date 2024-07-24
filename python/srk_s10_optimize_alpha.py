#%%
import jax.numpy as jnp
from jax import jit, value_and_grad, random, vmap
import optax
from tqdm import tqdm

from srkS10scalarnoise import srk_s10_scalar_noise
from duffingvanderpol import drift, diffusion


num_samples = 10
target_alpha = 1.0
seed = 42
target_key = random.PRNGKey(seed)
adam_lr = 0.01
alpha_opt_init = -1.0 # Initial value of alpha
num_opt_iters = 4000

opt = optax.adam(learning_rate=adam_lr)
opt_state = opt.init(alpha_opt_init)


#%%
ddkeys = random.split(random.PRNGKey(seed), num_samples)
init_xs = jnp.array([[-2 - 0.2*(j+1), 1] for j in range(num_samples)])
init_keys = random.split(random.PRNGKey(seed), num_samples)
#%%
t_init = 0
t_final = 20
step_size = 2**-5
num_srk_steps = int(t_final / step_size)
tspan = jnp.linspace(0, t_final, num_srk_steps)
init_x = init_xs[0]
target_trajectory = srk_s10_scalar_noise(target_key, drift, diffusion, tspan, init_x, 1.0, target_alpha)

def loss_fn(key, test_alpha):
    # Simulate test trajectory at each initial condition
    key, sample_key = random.split(key)
    init_x = random.normal(sample_key, shape=(2,))
    test_trajectory = srk_s10_scalar_noise(key, drift, diffusion, tspan, init_x, 1.0, test_alpha)
    mean_test = jnp.mean(test_trajectory)
    mean_target = jnp.mean(target_trajectory)
    # second moments also
    var_test = jnp.var(test_trajectory)
    var_target = jnp.var(target_trajectory)
    return (mean_test - mean_target)**2 + (var_test - var_target)**2

loss_and_grad = value_and_grad(loss_fn, argnums=1)

@jit
def opt_step(rng, opt_state, alpha):
    # keys = random.split(rng, num_samples)
    # loss, grads = vmap(loss_and_grad, (0, None))(keys, alpha)
    loss, grads = loss_and_grad(rng, alpha)
    updates, opt_state = opt.update(grads, opt_state)
    alpha = optax.apply_updates(alpha, updates)
    return loss, opt_state, alpha

alpha = alpha_opt_init
key = random.PRNGKey(0)
for i in tqdm(range(num_opt_iters)):
    key, subkey = random.split(key)
    loss, opt_state, alpha = opt_step(subkey, opt_state, alpha)
    if i % 200 == 0:
        print(f"Iteration {i}: Loss = {loss}, Alpha = {alpha}")


# %%
