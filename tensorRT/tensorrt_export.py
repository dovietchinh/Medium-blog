import os
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def onnx2engine():
    model_file = 'yolov5s.onnx'
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB

    flag = int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) 
    flag = (1 << flag)
    network = builder.create_network(flag) # network : 

    parser = trt.OnnxParser(network,logger)
    parser.parse_from_file(model_file)

    engine = builder.build_engine(network,config)

    with open("yolov5s.engine",'wb') as f:
        f.write(engine.serialize())
        
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * 2#engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def inference_engine():
    engine_file = "yolov5s.engine"
    
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)

    with open(engine_file,'rb') as f:
        data = f.read()
    engine = runtime.deserialize_cuda_engine(data)
    context = engine.create_execution_context()

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    img = cv2.imread("data/images/bus.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(640,640))
    img = np.transpose(img,[2,0,1])[None]
    img = img.astype('float32')/255.
    img = np.concatenate([img,img],axis=0)

    inputs[0].host = img.copy()

    output = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #pred = np.argmax(output[0])
    print(output[0].shape)

    

if __name__ =='__main__':
    inference_engine()
