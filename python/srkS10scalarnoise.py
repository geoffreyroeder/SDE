# function [x,tspan] = srkS10scalarnoise(f,L,tspan,x0,Q)
# %% SRKS10SCALARNOISE - Numerical SDE solver: Stochastic RK, strong 1.0
# %
# % Syntax:
# %   [x,tspan] = srkS10scalarnoise(f,L,tspan,x0,Q)
# %
# % In:
# %   f      - Drift function, f(x,t)
# %   L      - Diffusion function, L(x,t)
# %   tspan  - Time steps to simulate, [t0,...,tend]
# %   x0     - Initial condition
# %   Q      - Spectral density (default: standard Brownian motion)
# %
# % Out:
# %   x      - Solved values
# %   tspan  - Time steps
# %   
# % Description:
# %   Integrates the system of stochatic differential equations
# %     dx = f(x,t) dt + L(x,t) dbeta,  for x(0) = x0
# %   over the time interval defined in tspan.
# %
# % Copyright: 
# %   2018 - Simo Särkkä and Arno Solin
# %
# % License:
# %   This software is provided under the MIT License. See the accompanying 
# %   LICENSE file for details.

# %%

#   % Check if Q given
#   if nargin<5 || isempty(Q), Q = eye(size(L(x0,tspan(1)),2)); end 

#   % NB: Only for scalar beta
#   if size(Q,1)>1, error('NB: Only for scalar beta.'), end

#   % Number of steps
#   steps = numel(tspan);
  
#   % Allocate space
#   x = zeros(size(x0,1),steps);

#   % Initial state
#   x(:,1) = x0;
  
#   % Pre-calculate random numbers
#   R = randn(1,steps);
  
#   % Iterate
#   for k=2:steps

#     % Time discretization
#     dt = tspan(k)-tspan(k-1);

#     % Increment
#     db  = sqrt(dt*Q)*R(1,k);
#     dbb = 1/2*(db^2 - Q*dt);
        
#     % Evaluate only once
#     fx = f(x(:,k-1),tspan(k-1));
#     Lx = L(x(:,k-1),tspan(k-1));
    
#     % Supporting values
#     x2  = x(:,k-1) + fx*dt;
#     tx2 = x2 + Lx*dbb/sqrt(dt);
#     tx3 = x2 - Lx*dbb/sqrt(dt);
    
#     % Evaluate the remaining values
#     fx2 = f(x2,tspan(k-1)+dt);
#     Lx2 = L(tx2,tspan(k-1)+dt);
#     Lx3 = L(tx3,tspan(k-1)+dt);
    
#     % Step
#     x(:,k) = x(:,k-1) + ...
#         (fx+fx2)*dt/2 + ...
#         Lx*db + ...
#         sqrt(dt)/2*(Lx2 - Lx3);
    
#   end

import jax.numpy as jnp
from jax import jit, random, vmap
from functools import partial
import tqdm

@partial(jit, static_argnums=(1, 2))
def srk_s10_scalar_noise_step(rng, f, L, x, t, dt, Q):
    # Increment
    db = jnp.sqrt(dt * Q) * random.normal(rng, shape=(1,))
    dbb = 1 / 2 * (db ** 2 - Q * dt)

    # Evaluate only once
    fx = f(x, t)
    Lx = L(x, t)

    # Supporting values
    x2 = x + fx * dt
    tx2 = x2 + Lx * dbb / jnp.sqrt(dt)
    tx3 = x2 - Lx * dbb / jnp.sqrt(dt)

    # Evaluate the remaining values
    fx2 = f(x2, t + dt)
    Lx2 = L(tx2, t + dt)
    Lx3 = L(tx3, t + dt)

    # Step
    x_next = x + (fx + fx2) * dt / 2 + Lx * db + jnp.sqrt(dt) / 2 * (Lx2 - Lx3)
    return x_next

def srk_s10_scalar_noise(key, f, L, tspan, x0, Q=None):
    if Q is None:
        Q = 1.0  # Standard Brownian motion

    steps = len(tspan)
    x = jnp.zeros((steps, x0.shape[0]))
    x = x.at[0].set(x0)

    for i in tqdm.tqdm(range(steps - 1)):
        key, subkey = random.split(key)
        dt = tspan[i + 1] - tspan[i]
        x = x.at[i + 1].set(srk_s10_scalar_noise_step(subkey, f, L, x[i], tspan[i], dt, Q))

    return x

if __name__ == "__main__":
    from duffingvanderpol import f, L

    tspan = jnp.linspace(0, 20, int(20 / (2**-5)))
    x0s = jnp.array([[-2 - 0.2*(j+1), 0] for j in range(10)])
    print(x0s.shape)
    #%%
    keys = random.split(random.PRNGKey(0), 10)
    xs = vmap(srk_s10_scalar_noise, (None, None, None, None, 0))(keys[0], f, L, tspan, x0s)
    #%%
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for j in range(10):
        x = xs[j, :, 0]
        y = xs[j, :, 1]
        u = f(jnp.stack([x, y]), tspan)
        u_x = u[0]
        u_y = u[1]
        # ax.quiver(x[15::25], y[15::25], u_x[15::25], u_y[15::25], angles='xy', scale_units='xy', width=0.005, headlength=5, headaxislength=5, alpha=0.5, scale=20)
        ax.plot(x, y, '-k', lw=0.5)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-4.8, 10)
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_aspect('equal')
  
    plt.tight_layout()
    plt.show()

# %%
