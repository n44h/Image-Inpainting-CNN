import torch


class ImageInpaintingCNN(torch.nn.Module):
    def __init__(self, num_input_channels: int = 4, num_hidden_layers: int = 5,
                 num_kernels: int = 64, kernel_size: int = 3):
        super().__init__()

        # List to add NN layers.
        layers = []

        """
        One CNN block consists of 2 parts:
        1. Convolution layer
        2. Activation function (ReLU)
        """

        # Loop to create N number of CNN blocks, where N = num_hidden_layers.
        for i in range(num_hidden_layers):
            # Add a 2D Convolutional layer.
            layers.append(
                torch.nn.Conv2d(
                    in_channels=num_input_channels,
                    out_channels=num_kernels,
                    kernel_size=kernel_size,
                    padding=int(kernel_size / 2)
                )
            )

            # Add a ReLU activation function.
            layers.append(
                torch.nn.ReLU()
            )

            num_input_channels = num_kernels

        # Compile the CNN blocks above into one sequential layer.
        self.hidden_layers = torch.nn.Sequential(*layers)

        # Creating output layer. Using Convolution layer.
        self.output_layer = torch.nn.Conv2d(in_channels=num_input_channels,
                                            out_channels=3,
                                            kernel_size=kernel_size,
                                            padding=int(kernel_size / 2))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        cnn_out = self.hidden_layers(input_tensor)
        predictions = self.output_layer(cnn_out)
        return predictions
