#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import neural_filters
import numpy as np
import torch


def theta_to_modulus(thetas, fs=200):
    return np.exp(-1 / (thetas * fs))


def modulus_to_theta(poles, fs=200):
    return - 1 / (fs * np.log(poles))


class BaseModel(torch.nn.Module):
    def __init__(self, thetas):
        super().__init__()

        self.thetas = thetas
        self.filters = None
        self.register_buffer('normalisation_weights', torch.Tensor([38.43190559738741, -50.05233847007584, 25.07626762013403, 3.1930363795157106]).unsqueeze_(1))
        self.register_buffer('normalisation_bias', torch.Tensor([48.95299158714191]))

    def forward(self, x, sum_filters=True):
        out = self.filters(x)
        modulus = out[2]

        modulus_tensor = modulus.new(modulus.size(0), 4)
        modulus_tensor[:, 0] = modulus
        modulus_tensor[:, 1] = modulus.exp()
        modulus_tensor[:, 2:] = modulus_tensor[:, :2].pow(2)

        modulus_tensor = modulus.new(modulus.size(0), 2)
        modulus_tensor[:, 0] = modulus
        modulus_tensor[:, 1] = modulus.exp()

        modulus_tensor = torch.cat((modulus_tensor, modulus_tensor[:, :2].pow(2)), -1)

        normalisation = torch.addmm(self.normalisation_bias, modulus_tensor, self.normalisation_weights)
        normalisation = normalisation.view(1, 1, -1)

        out = out[0]
        if isinstance(out, torch.nn.utils.rnn.PackedSequence):
            out, sizes = torch.nn.utils.rnn.pad_packed_sequence(out)

        normalized_out = out * normalisation
        return normalized_out.sum(dim=-1, keepdim=True) if sum_filters else normalized_out

    def filters_forward(self, x):
        """Get output of each filter without their superposition."""
        return self.forward(x, sum_filters=False)
        # out = self.filters(x)
        # modulus = out[2]
        #
        # modulus_tensor = modulus.new(modulus.size(0), 4)
        # modulus_tensor[:, 0] = modulus
        # modulus_tensor[:, 1] = modulus.exp()
        # modulus_tensor[:, 2:] = modulus_tensor[:, :2].pow(2)
        #
        # modulus_tensor = modulus.new(modulus.size(0), 2)
        # modulus_tensor[:, 0] = modulus
        # modulus_tensor[:, 1] = modulus.exp()
        #
        # modulus_tensor = torch.cat((modulus_tensor, modulus_tensor[:, :2].pow(2)), -1)
        #
        # normalisation = torch.addmm(self.normalisation_bias, modulus_tensor, self.normalisation_weights)
        # normalisation = normalisation.view(1, 1, -1)
        #
        # out = out[0]
        # if isinstance(out, torch.nn.utils.rnn.PackedSequence):
        #     out, sizes = torch.nn.utils.rnn.pad_packed_sequence(out)
        #
        # normalized_out = out * normalisation
        # return normalized_out


class ComplexModel(BaseModel):
    def __init__(self, thetas, phase_init=0):
        super().__init__(thetas)

        self.filters = neural_filters.NeuralFilter2CC(thetas.size)
        self.filters.reset_parameters(init_theta=phase_init, init_modulus=theta_to_modulus(thetas))

    def reset(self, thetas=None):
        if thetas is None:
            thetas = self.thetas
        self.filters.reset_parameters(init_theta=0, init_modulus=theta_to_modulus(thetas))


class CriticalModel(BaseModel):
    def __init__(self, thetas):
        super().__init__(thetas)

        self.filters = neural_filters.NeuralFilter2CD(thetas.size)
        self.filters.reset_parameters(init=theta_to_modulus(thetas))

    def reset(self, thetas=None):
        if thetas is None:
            thetas = self.thetas

        self.filters.reset_parameters(init=theta_to_modulus(thetas))
