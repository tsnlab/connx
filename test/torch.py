import torch
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        
        self.block_1 = torch.nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        x = self.block_1(x)
        return x

x = torch.randn(16, 3, 100, 250)
model = Model1()
model.eval()
model_onnx = torch.onnx._export(model, x, "conv2d.onnx",
                                verbose=True,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                                example_outputs=model(x),
                               )
model_onnx
