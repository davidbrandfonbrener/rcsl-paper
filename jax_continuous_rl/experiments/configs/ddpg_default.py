import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 2

    config.policy_noise = 0.2
    config.noise_clip = 0.5

    config.bc_alpha = 2.5

    return config
