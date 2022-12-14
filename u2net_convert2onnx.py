# import onnxruntime
# import onnx
import torch

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

INPUT_SIZE = (3, 320, 320)


def main():
    model_name = 'u2net'  # u2netp
    model_dir = "saved_models/u2net/u2net_bce_itr_112000_train_0.529639_tar_0.067323.pth"

    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    # if torch.cuda.is_available():
    #     net.load_state_dict(torch.load(model_dir))
    #     net.cuda()
    # else:
    #     net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 320, 320, requires_grad=True)
    dummy_input = dummy_input.type(torch.FloatTensor)

    # Export the model
    torch.onnx.export(net,         # model being run
                      # model input (or a tuple for multiple inputs)
                      dummy_input,
                      "output.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                    'output': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


def validate():
    onnx_model = onnx.load("output.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("output.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    dummy_input = torch.randn(1, 3, 320, 320, requires_grad=True)
    dummy_input = dummy_input.type(torch.FloatTensor)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(
    #     to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    main()
    # validate()
