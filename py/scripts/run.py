
import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax


def init_network_params(layer_sizes, key):
    """Initialize network parameters."""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for in_size, out_size, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        w_key, b_key = jax.random.split(k)
        params.append({
            'weights': jax.random.normal(w_key, (in_size, out_size)) * 0.1,
            'bias': jax.random.normal(b_key, (out_size,)) * 0.1
        })
    return params


@jit
def forward(params, x):
    """Forward pass through the network."""
    activation = x
    for layer in params:
        activation = jnp.dot(activation, layer['weights']) + layer['bias']
        activation = jax.nn.relu(activation)
    return activation


@jit
def loss_fn(params, x, y):
    """Mean squared error loss."""
    pred = forward(params, x)
    return jnp.mean((pred - y) ** 2)


@jit(static_argnames=['optimizer'])
def update(params, x, y, optimizer_state, optimizer):
    """Perform one optimization step."""
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss


async def main():
    # Generate synthetic data
    key = random.PRNGKey(0)
    x_key, y_key, model_key = random.split(key, 3)
    
    x = random.normal(x_key, (100, 2))
    y = jnp.sum(x ** 2, axis=1, keepdims=True)
    
    # Initialize network
    layer_sizes = [2, 32, 32, 1]
    params = init_network_params(layer_sizes, model_key)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate=0.01)
    optimizer_state = optimizer.init(params)
    
    # Training loop
    for i in range(1000):
        params, optimizer_state, loss = update(params, x, y, optimizer_state, optimizer)
        if i % 100 == 0:
            print(f"Step {i}, Loss: {loss:.4f}")
    
    # Test prediction
    test_x = jnp.array([[0.5, 0.5]])
    pred = forward(params, test_x)
    print(f"\nTest input: {test_x}")
    print(f"Predicted: {pred[0,0]:.4f}")
    print(f"Expected: {jnp.sum(test_x ** 2):.4f}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
