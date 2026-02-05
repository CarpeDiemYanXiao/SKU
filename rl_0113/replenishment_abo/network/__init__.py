from .PolicyNetwork import PolicyNetwork
from .PolicyNetwork_continue_action import PolicyNetwork_continue_action
from .ValueNetwork import ValueNetwork
from .QNetwork import QNetwork
from .base_feature_extractor import flatten_feature_extractor
networks_dict = {
    "PolicyNetwork": PolicyNetwork,
    "ValueNetwork": ValueNetwork,
    "PolicyNetwork_continue_action":PolicyNetwork_continue_action,
    "Base_flatten_feature_extractor": flatten_feature_extractor,
    "QNetwork":QNetwork
}