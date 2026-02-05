from .base_loss import Base_loss
from .dqn_loss import dqn_loss

loss_dict = {
    "Base_loss": Base_loss,
    "dpn_loss": dqn_loss
}
