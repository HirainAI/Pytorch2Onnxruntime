import onnx
import numpy as np
import onnxruntime as rt
import time

model_path = './model.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

sess = rt.InferenceSession(model_path)
# sess.set_providers(["TensorrtExecutionProvider"])
# sess.set_providers(["CPUExecutionProvider"])
sess.set_providers(["CUDAExecutionProvider"])

image = cv2.imread("hrnet_demo.jpg")
image = cv2.resize(image, (288,384))
image = image.astype(np.float32)/255.0

image = image.transpose(2,0,1)
image = np.array(image)[np.newaxis, :, :, :]
print(image.shape)

input_name_1 = sess.get_inputs()[0].name
output_name_1 = sess.get_outputs()[0].name
output_name_2 = sess.get_outputs()[1].name

print("input_name_1:",input_name_1)
print("output_name_1:",output_name_1)
print("output_name_2:",output_name_2)

i=0
while i<10:
    start = time.time()
    output = sess.run([output_name_1,output_name_2], {(input_name_1): image})
    print('spend time:',(time.time()-start)*1000.0)

    i+=1
————————————————
版权声明：本文为CSDN博主「脆皮茄条」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_43917589/article/details/123049323