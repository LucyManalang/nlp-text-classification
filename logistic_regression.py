import torch
import torch.nn as nn
from typing import Mapping

class LogisticRegression(nn.Module): # used https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html as a reference
    def __init__(self, input_size : int, num_classes : int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def train_model(self, x_train: torch.Tensor, y_train: torch.Tensor, epochs : int = 100, loss: float = 0.01) -> None: # based off of stochastic gradient descent in Jurafsky & Martin
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            #forward pass
            logits = self(x_train)
            loss = loss_fn(logits, y_train)

            #backward pass
            self.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in self.parameters():
                    param -= loss * param.grad

            #for debugging
            # if epoch % 10 == 0:
            #     print("Epoch {} | Loss: {:.4}".format(epoch, loss))