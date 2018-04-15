from tensorflow.python.tools import inspect_checkpoint as chkp
#chkp.print_tensors_in_checkpoint_file("./frozen_graph/my_test_model.ckpt", tensor_name='', all_tensors=True,all_tensor_names=False)
chkp.print_tensors_in_checkpoint_file("./frozen_graph/my_test_model-1000", tensor_name='', all_tensors=True,all_tensor_names=False)
