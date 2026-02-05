from .base_ppo import Base_ppo, ppo_continue_action,Base_ppoConfig

from .base_dqn import Base_dqn, Base_dqnConfig
from .feature_extractor_ppo import feature_extractor_ppo, feature_extractor_ppoConfig
model_dict = {
    "base_ppo": Base_ppo,
    "ppo_continue_action":ppo_continue_action,
    "feature_extractor_ppo":feature_extractor_ppo,
    "base_dqn": Base_dqn
}

modelconfig_dict = {
    "base_ppo": Base_ppoConfig,
    "ppo_continue_action":Base_ppoConfig,
    'base_dqn': Base_dqnConfig,
    "feature_extractor_ppo":feature_extractor_ppoConfig,
}
