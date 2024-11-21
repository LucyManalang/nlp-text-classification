import torch
import torch.nn as nn
from typing import Mapping

class LogisticRegression(nn.Module): # used https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html as a reference
    def __init__(self, input_size : int, classes : set[int], vocab : Mapping[str, int], vector : torch.Tensor):
        self.classes = classes
        self.vocab = vocab
        self.vector = vector

        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, len(classes))
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    
    def train_logistic_regression(self, epochs : int = 100, loss: float = 0.01) -> None: # based off of stochastic gradient descent in Jurafsky & Martin
        x = self.vector 
        y = torch.tensor([c for c in self.classes], dtype=torch.long) # class labels

        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            logits = self(x)
            loss = loss_fn(logits, y)

            self.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in self.parameters():
                    param -= loss * param.grad

            # if epoch % 10 == 0:
            #     print("Epoch {} | Loss: {:.4}".format(epoch, loss))