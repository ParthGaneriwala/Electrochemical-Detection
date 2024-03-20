from data.src.model import LeNet

model = LeNet()
print(model)

for name, param in model.named_parameters():
    print(name, param.size(), param.requires_grad)