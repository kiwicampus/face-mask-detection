import os
import time
import typing as tp
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image

# matplotlib.use('TkAgg')




class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine, batch_size=1):
    """Allocates host and device buffer for TRT engine inference.
    This function is similair to the one in common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {
        "input_1": np.float32,
        "output_bbox/BiasAdd": np.float32,
        "output_cov/Sigmoid": np.float32,
    }

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dtype = binding_to_type[str(binding)]
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


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# When infering on single image, we measure inference
# time to output it to the user
inference_start_time = time.time()

# Output inference time
print(
    "TensorRT inference time: {} ms".format(
        int(round((time.time() - inference_start_time) * 1000))
    )
)


# -------------- MODEL PARAMETERS FOR DETECTNET_V2 --------------------------------
class DetectNetV2(object):
    def __init__(
        self,
        engine_path: str,
        num_classes: int,
        input_shape: tp.List[int] = [544, 960],
        batch_size: int = 1,
        stride: int = 16,
        box_norm: float = 35.0,
        dla_core: int = None,
        gpu_fallback: bool = False,
    ):

        # Load DLA configs if desired
        if dla_core is not None:
            trt.IBuilderConfig.DLA_core = dla_core
            trt.BuilderFlag.GPU_FALLBACK = gpu_fallback
            trt.IBuilderConfig.default_device_type = trt.DeviceType.DLA

        # TensorRT logger singleton
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Tensorrt Engine creation
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = load_engine(self.trt_runtime, engine_path)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.batch_size = batch_size
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.trt_engine, batch_size=self.batch_size
        )

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        self.model_h = input_shape[0]
        self.model_w = input_shape[1]
        self.stride = stride
        self.box_norm = box_norm
        self.num_classes = num_classes

        # Calculate parameters for postprocessing
        self.grid_h = int(self.model_h / stride)
        self.grid_w = int(self.model_w / stride)
        self.grid_size = self.grid_h * self.grid_w

        self.grid_centers_w = []
        self.grid_centers_h = []

        for i in range(self.grid_h):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_h.append(value)

        for i in range(self.grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            self.grid_centers_w.append(value)

    def applyBoxNorm(self, o1, o2, o3, o4, x, y):
        """
        Applies the GridNet box normalization
        Args:
            o1 (float): first argument of the result
            o2 (float): second argument of the result
            o3 (float): third argument of the result
            o4 (float): fourth argument of the result
            x: row index on the grid
            y: column index on the grid

        Returns:
            float: rescaled first argument
            float: rescaled second argument
            float: rescaled third argument
            float: rescaled fourth argument
        """
        o1 = (o1 - self.grid_centers_w[x]) * -self.box_norm
        o2 = (o2 - self.grid_centers_h[y]) * -self.box_norm
        o3 = (o3 + self.grid_centers_w[x]) * self.box_norm
        o4 = (o4 + self.grid_centers_h[y]) * self.box_norm
        return o1, o2, o3, o4

    def postprocess(self, outputs, min_confidence, analysis_classes, wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        bbs = []
        class_ids = []
        scores = []
        for c in analysis_classes:

            x1_idx = c * 4 * self.grid_size
            y1_idx = x1_idx + self.grid_size
            x2_idx = y1_idx + self.grid_size
            y2_idx = x2_idx + self.grid_size

            boxes = outputs[0]
            for h in range(self.grid_h):
                for w in range(self.grid_w):
                    i = w + h * self.grid_w
                    score = outputs[1][c * self.grid_size + i]
                    if score >= min_confidence:
                        o1 = boxes[x1_idx + w + h * self.grid_w]
                        o2 = boxes[y1_idx + w + h * self.grid_w]
                        o3 = boxes[x2_idx + w + h * self.grid_w]
                        o4 = boxes[y2_idx + w + h * self.grid_w]

                        o1, o2, o3, o4 = self.applyBoxNorm(o1, o2, o3, o4, w, h)

                        xmin = int(o1)
                        ymin = int(o2)
                        xmax = int(o3)
                        ymax = int(o4)
                        if wh_format:
                            bbs.append([xmin, ymin, xmax - xmin, ymax - ymin])
                        else:
                            bbs.append([xmin, ymin, xmax, ymax])
                        class_ids.append(c)
                        scores.append(float(score))

        return bbs, class_ids, scores

    def preprocess(self, image: np.ndarray):
        image = cv2.resize(image, (self.model_w, self.model_h))
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        # Normalize to [0.0, 1.0] interval (expected by model)
        image = (1.0 / 255.0) * image
        return image

    def predict(
        self, image: np.ndarray, threshold: float = 0.1, nms_threshold: float = 0.5
    ):
        """Infers model on batch of same sized images resized to fit the model.
        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        orig_h, orig_w = image.shape[:2]
        image = self.preprocess(image)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, image.ravel())

        # Fetch output from the model
        [detection_out, keepCount_out] = do_inference(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream,
        )

        bboxes, class_ids, scores = self.postprocess(
            [detection_out, keepCount_out], threshold, range(self.num_classes)
        )

        # Final bboxes only after NMS
        indexes = cv2.dnn.NMSBoxes(bboxes, scores, threshold, nms_threshold)

        # Get final structure
        detections = []
        for idx in indexes:
            idx = int(idx)
            xmin, ymin, w, h = bboxes[idx]

            # check if we need to rescale bboxes
            if [orig_h, orig_w] != [self.model_h, self.model_w]:
                xmin_rel = xmin/self.model_w
                ymin_rel = ymin/self.model_h
                w_rel = w/self.model_w
                h_rel = h/self.model_h
                # convert to new dimensions
                xmin = int(xmin_rel * orig_w)
                ymin = int(ymin_rel * orig_h)
                w = int(w_rel * orig_w)
                h = int(h_rel * orig_h)

            class_id = class_ids[idx]
            score = scores[idx]
            detections.append(
                {"bbox": [xmin, ymin, w, h], "class_id": class_id, "score": score}
            )
        return detections


NUM_CLASSES = 2
threshold = 0.25
model = DetectNetV2(
    "detectnet_v2/experiment_dir_final_rt/resnet18_detector.trt", NUM_CLASSES
)

image = cv2.cvtColor(cv2.imread("6000-all-kitti-format/test_images/2020-12-11-172854.jpg"), cv2.COLOR_BGR2RGB)
detections = model.predict(image, threshold)

for det in detections:
    xmin, ymin, w, h = det["bbox"]
    class_id = det["class_id"]
    color = [0, 0, 255] if class_id else [255, 0, 0]
    cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, 2)

# plt.imshow(image)
# plt.show()

# cap =  cv2.VideoCapture(0)
# cap =  cv2.VideoCapture("berkeley2019.mp4")
input_video = "mask_test.mp4"
cap =  cv2.VideoCapture(input_video)
file_writer = None
output_path = Path(input_video).stem + "faces.mp4"

while cap.isOpened():
    captured, image = cap.read()
    if not captured:
        break

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = model.predict(image, threshold)

    for det in detections:
        xmin, ymin, w, h = det["bbox"]
        class_id = det["class_id"]
        color = [0, 0, 255] if class_id else [255, 0, 0]
        cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color, 2)
    
    cv2.imshow("detections", image[...,::-1])
    cv2.waitKey(1)

    if file_writer is None and output_path is not None:
        file_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"MP4V"),
            5,  # int(fps),
            image.shape[:2][::-1],
        )
    if file_writer is not None:
        file_writer.write(image[...,::-1])

cap.release()
if file_writer is not None:
    file_writer.release()
