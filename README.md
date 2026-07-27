# ncnn-arm64-yolo11n
Set up the NCNN environment in the Ubuntu virtual machine and cross-compile the .param and .bin files of the YOLO11 model trained by yourself into FP16 or INT8 types. Deploy them onto the Orangepi Zero 2 1G development board to recognize images.

Ubuntu virtual machine version：
---------------------------------------
	hkd@hkd-virtual-machine:~/Downloads/ncnn/build-cpu$ lsb_release -a
	No LSB modules are available.
	Distributor ID: Ubuntu
	Description:    Ubuntu 22.04.5 LTS
	Release:        22.04
	Codename:       jammy

Orangepi Zero 2version：
---------------------------------------
	orangepi@orangepizero2:~/Downloads/drink_demo$ lsb_release -a
	No LSB modules are available.
	Distributor ID: Ubuntu
	Description:    Ubuntu 22.04.2 LTS
	Release:        22.04
	Codename:       jammy

一、Setup of Ubuntu virtual machine environment (performed according to the official user manual of Orangepi Zero 2)
---------------------------------------
1、Download the source code：
---------------------------------------
	git clone https://github.com/Tencent/ncnn.git

2、Install the required packages：
---------------------------------------
	sudo apt update
	sudo apt install build-essentialgitcmake libprotobuf-dev protobuf-compiler libopencv-dev

3、Disable the benchncnn-related model tests in the annotations：
---------------------------------------
It is up to personal preference to decide whether to block it. Since the ncnn file downloaded from the official manual is not the latest version, the official only commented on blocking the vgg16 model for testing.

	//benchmark("alexnet", ncnn::Mat(227, 227, 3), opt, alexnet_param_data);
	//benchmark("vgg16", ncnn::Mat(224, 224, 3), opt, vgg16_param_data);
	//benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt, vgg16_int8_param_data);
	//benchmark("resnet50", ncnn::Mat(224, 224, 3), opt, resnet50_param_data);
	//benchmark("resnet50_int8", ncnn::Mat(224, 224, 3), opt, resnet50_int8_param_data);


4、Compilation environment：
---------------------------------------
	mkdir build
	cd build
Before using CMake, please make sure that you have the cross-compilation tools for the OrangePi Zero 2.
The version of gcc I'm using is gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.
And add the environment variable instructions in ~/.bashrc (please modify according to your own path):
export PATH=/home/$(whoami)/Downloads/orangepi-build-next/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin:$PATH

	cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_EXAMPLES=ON
	make -j$(nproc)

5、Test the inference performance of the neural network：
---------------------------------------
Detailed operations can be carried out by viewing the official user manual of Orangepi Zero 2.
Transfer the entire benchmark from the Ubuntu virtual machine to the development board via the SCP command for testing.


6、It is recommended to create another folder named "build-x86" (you can customize the name) under the "ncnn" folder directory.
---------------------------------------
Since we plan to convert the model to int8 type .param and .bin files in the later stage,
The ncnn2table and ncnn2int8 executable files need to be used on a PC or an Ubuntu virtual machine.
The previously created "build" folder is specifically designed for the arm64 architecture.
The parameter -DNCNN_BUILD_TOOLS was not added, which resulted in the absence of the "tools" folder being generated under the "build" folder.
And it is also unlikely to be compiled on the development board.
There is no need to add any specific compilation tools. The default one is the x86 compilation tool.

	mkdir build-x86
	cd build-x86
	cmake -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_TOOLS=ON
	make -j$(nproc)
After the setup is completed, you can go to /build-x86/tools/quantize to check if there are executable files named "ncnn2table" and "ncnn2int8".
If it is not possible to compile manually



二、Model Conversion to FP16 Format (It is highly recommended to follow the comments and operation steps in the official yolo11.cpp source file of NCNN. Otherwise, it will not be possible to directly match and apply the official source file!!!).If you use the one-click conversion tool provided by the Ultralytics website to convert the file format to NCNN, you will need to modify the official yolo11.cpp source code of NCNN yourself. Moreover, this method cannot achieve the optimization benefits of NCNN.You can directly operate by opening the commented content in the official yolo11.cpp file: https://github.com/Tencent/ncnn/blob/master/examples/yolo11.cpp.By default, param and.bin files in FP16 format are generated.
---------------------------------------
1、Set up the environment (using PC Anaconda Prompt terminal)
---------------------------------------
	conda activate (Environmental Name)
	pip3 install -U ultralytics pnnx ncnn

2、export yolo11 torchscript(PC Anaconda Prompt terminal operation)
---------------------------------------
Since my dataset is not from the official source, but rather a dataset that I trained myself to recognize 8 types of quantities.
In the Anaconda Prompt terminal of the PC, use the command `cd` to enter the folder where you have customized the model file named `best.pt`, and rename `best.pt` to `yolo11n.pt`.

	yolo export model=yolo11n.pt format=torchscript

3、convert torchscript with static shape(PC Anaconda Prompt terminal operation)
---------------------------------------

	pnnx yolo11n.torchscript

4、modify yolo11n_pnnx.py for dynamic shape inference(PC Anaconda Prompt terminal operation)
---------------------------------------
	//		A. modify reshape to support dynamic image sizes
	//      B. permute tensor before concat and adjust concat axis
	//      C. drop post-process part
	//      before:
	//          v_235 = v_204.view(1, 144, 6400)
	//          v_236 = v_219.view(1, 144, 1600)
	//          v_237 = v_234.view(1, 144, 400)
	//          v_238 = torch.cat((v_235, v_236, v_237), dim=2)
	//          ...
	//      after:
	//          v_235 = v_204.view(1, 144, -1).transpose(1, 2)
	//          v_236 = v_219.view(1, 144, -1).transpose(1, 2)
	//          v_237 = v_234.view(1, 144, -1).transpose(1, 2)
	//          v_238 = torch.cat((v_235, v_236, v_237), dim=1)
	//          return v_238
Modify the content of your own yolo11n_pnnx.py. Below is the modification I made to the yolo11n_pnnx.py that I generated myself.
It was only through AI-assisted query that I discovered that the processing method in the official yolo11n_pnnx.py was different from the way my PC generated yolo11n_pnnx.py.
Moreover, the official has simplified the operation in this part. In fact, the modified code v_238 = torch.cat((v_235, v_236, v_237), dim=1) needs to be implemented.
Delete or comment out all the following codes, and then add the "return v_238" code.
However, the final results are all the same. For beginners like me who don't have much knowledge, we usually solve the problems by querying through AI. If there are any issues, we can solve them by contacting AI.
		
		v_187 = self.model_22_cv2_conv(v_186)
        v_188 = self.pnnx_unique_59(v_187)
        v_189 = self.pnnx_188_data
        v_190 = self.model_23_cv2_0_0_conv(v_146)
        v_191 = self.pnnx_unique_60(v_190)
        v_192 = self.model_23_cv2_0_1_conv(v_191)
        v_193 = self.pnnx_unique_61(v_192)
        v_194 = self.model_23_cv2_0_2(v_193)
        # v_195 = v_194.reshape(1, 64, 6400)
        v_195 = v_194.flatten(2)
        v_196 = self.model_23_cv2_1_0_conv(v_161)
        v_197 = self.pnnx_unique_62(v_196)
        v_198 = self.model_23_cv2_1_1_conv(v_197)
        v_199 = self.pnnx_unique_63(v_198)
        v_200 = self.model_23_cv2_1_2(v_199)
        # v_201 = v_200.reshape(1, 64, 1600)
        v_201 = v_200.flatten(2)
        v_202 = self.model_23_cv2_2_0_conv(v_188)
        v_203 = self.pnnx_unique_64(v_202)
        v_204 = self.model_23_cv2_2_1_conv(v_203)
        v_205 = self.pnnx_unique_65(v_204)
        v_206 = self.model_23_cv2_2_2(v_205)
        # v_207 = v_206.reshape(1, 64, 400)
        v_207 = v_206.flatten(2)
        v_208 = torch.cat((v_195, v_201, v_207), dim=-1)
        v_209 = self.model_23_cv3_0_0_0_conv(v_146)
        v_210 = self.pnnx_unique_66(v_209)
        v_211 = self.model_23_cv3_0_0_1_conv(v_210)
        v_212 = self.pnnx_unique_67(v_211)
        v_213 = self.model_23_cv3_0_1_0_conv(v_212)
        v_214 = self.pnnx_unique_68(v_213)
        v_215 = self.model_23_cv3_0_1_1_conv(v_214)
        v_216 = self.pnnx_unique_69(v_215)
        v_217 = self.model_23_cv3_0_2(v_216)
        # v_218 = v_217.reshape(1, 8, 6400)
        v_218 = v_217.flatten(2)
        v_219 = self.model_23_cv3_1_0_0_conv(v_161)
        v_220 = self.pnnx_unique_70(v_219)
        v_221 = self.model_23_cv3_1_0_1_conv(v_220)
        v_222 = self.pnnx_unique_71(v_221)
        v_223 = self.model_23_cv3_1_1_0_conv(v_222)
        v_224 = self.pnnx_unique_72(v_223)
        v_225 = self.model_23_cv3_1_1_1_conv(v_224)
        v_226 = self.pnnx_unique_73(v_225)
        v_227 = self.model_23_cv3_1_2(v_226)
        # v_228 = v_227.reshape(1, 8, 1600)
        v_228 = v_227.flatten(2)
        v_229 = self.model_23_cv3_2_0_0_conv(v_188)
        v_230 = self.pnnx_unique_74(v_229)
        v_231 = self.model_23_cv3_2_0_1_conv(v_230)
        v_232 = self.pnnx_unique_75(v_231)
        v_233 = self.model_23_cv3_2_1_0_conv(v_232)
        v_234 = self.pnnx_unique_76(v_233)
        v_235 = self.model_23_cv3_2_1_1_conv(v_234)
        v_236 = self.pnnx_unique_77(v_235)
        v_237 = self.model_23_cv3_2_2(v_236)
        # v_238 = v_237.reshape(1, 8, 400)
        v_238 = v_237.flatten(2)
        v_239 = torch.cat((v_218, v_228, v_238), dim=-1)
        # === The following is the revised part.===
        # Concatenate the 64-dimensional Box features (v_208) and the 8-dimensional Cls features (v_239) to form [1, 72, 8400]
        v_raw = torch.cat((v_208, v_239), dim=1)
        # Transposed to [1, 8400, 72]
        v_out = v_raw.transpose(1, 2)
        return v_out
---------------------------------------
	//			D. modify area attention for dynamic shape inference
	//      before:
	//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
	//          v_96 = v_95.view(1, 2, 128, 400)
	//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
	//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
	//          v_101 = torch.matmul(input=v_100, other=v_98)
	//          v_102 = (v_101 * 0.176777)
	//          v_103 = F.softmax(input=v_102, dim=-1)
	//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
	//          v_105 = torch.matmul(input=v_99, other=v_104)
	//          v_106 = v_105.view(1, 128, 20, 20)
	//          v_107 = v_99.reshape(1, 128, 20, 20)
	//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
	//          v_109 = (v_106 + v_108)
	//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
	//      after:
	//          v_95 = self.model_10_m_0_attn_qkv_conv(v_94)
	//          v_96 = v_95.view(1, 2, 128, -1)
	//          v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
	//          v_100 = torch.transpose(input=v_97, dim0=-2, dim1=-1)
	//          v_101 = torch.matmul(input=v_100, other=v_98)
	//          v_102 = (v_101 * 0.176777)
	//          v_103 = F.softmax(input=v_102, dim=-1)
	//          v_104 = torch.transpose(input=v_103, dim0=-2, dim1=-1)
	//          v_105 = torch.matmul(input=v_99, other=v_104)
	//          v_106 = v_105.view(1, 128, v_95.size(2), v_95.size(3))
	//          v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))
	//          v_108 = self.model_10_m_0_attn_pe_conv(v_107)
	//          v_109 = (v_106 + v_108)
	//          v_110 = self.model_10_m_0_attn_proj_conv(v_109)
	
		# v_96 = v_95.reshape(1, 2, 128, 400)
        # v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32,32,64))
        # v_100 = (v_97 * 0.176776692)
        # v_101 = torch.transpose(v_100, dim0=-2, dim1=-1)
        # v_102 = torch.matmul(v_101, other=v_98)
        # v_103 = F.softmax(v_102, dim=-1)
        # v_104 = torch.transpose(v_103, dim0=-2, dim1=-1)
        # v_105 = torch.matmul(v_99, other=v_104)
        # v_106 = v_105.reshape(1, 128, 20, 20)
        # v_107 = v_99.reshape(1, 128, 20, 20)
        v_96 = v_95.reshape(1, 2, 128, -1)
        v_97, v_98, v_99 = torch.split(tensor=v_96, dim=2, split_size_or_sections=(32, 32, 64))
        v_100 = (v_97 * 0.176776692)
        v_101 = torch.transpose(v_100, dim0=-2, dim1=-1)
        v_102 = torch.matmul(v_101, other=v_98)
        v_103 = F.softmax(v_102, dim=-1)
        v_104 = torch.transpose(v_103, dim0=-2, dim1=-1)
        v_105 = torch.matmul(v_99, other=v_104)
        v_106 = v_105.reshape(1, 128, v_95.size(2), v_95.size(3))
        v_107 = v_99.reshape(1, 128, v_95.size(2), v_95.size(3))

5、re-export yolo11 torchscript(PC Anaconda Prompt terminal operation)
---------------------------------------
	python -c "import yolo11n_pnnx; yolo11n_pnnx.export_torchscript()"

6、convert new torchscript with dynamic shape(PC Anaconda Prompt terminal operation)
---------------------------------------
	pnnx yolo11n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]

7、now you get ncnn model files(PC Anaconda Prompt terminal operation)
---------------------------------------
	mv yolo11n_pnnx.py.ncnn.param yolo11n.ncnn.param
	mv yolo11n_pnnx.py.ncnn.bin yolo11n.ncnn.bin

8、Modify the official yolo11.cpp source file in the Ubuntu virtual machine (if not using the official dataset of YOLO)
---------------------------------------
By opening ncnn/
It is possible to manually disable the use of Vulkan acceleration. According to the AI query of the Zhi H616 chip, it is known that it does not support this feature. Additionally, I have personally tested it and found that it failed.
Here is the discussion about this issue: https://github.com/Tencent/ncnn/issues/6684
	
	yolo11.opt.use_vulkan_compute = false;
Modify the target size, as dynamic adjustment of the image input size has been set above. The smaller the target size, the less computational effort is required.

	const int target_size = (640 or 416 or 320);
Modify the type name of the identification. It can be updated by matching and modifying according to the "names" in the yaml file of the dataset.

	static const char* class_names[] = {}

9、Compile the source file YOLO11.cpp
---------------------------------------
Make the compilation in the "ncnn/build" directory, and it will automatically overwrite the existing "yolo11" executable file in the "examples" folder with the latest version.

	make yolo11

10、Conduct tests on the development board
---------------------------------------
Make sure that the .param and .bin files transferred from the PC end and the yolo11 executable file transferred from the Ubuntu virtual machine are placed in the same directory on the development board.
At the same time, this path requires an image as an input parameter for the yolo11 executable file.
The following is a simple multiple test observation of a single run using the "time" command. The input images are respectively 640x640, 416x416, and 320x320.
On average, how much time is required：

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_640f16 640.jpeg
	3 = 0.96976 at 0.02 324.91 83.14 x 303.49
	3 = 0.96807 at 532.37 321.49 106.63 x 316.62
	3 = 0.96587 at 441.40 321.01 107.13 x 312.01
	3 = 0.96315 at 160.31 329.88 85.30 x 308.38
	3 = 0.95825 at 242.96 328.38 89.23 x 310.62
	3 = 0.95739 at 77.23 327.57 87.55 x 307.80
	3 = 0.95169 at 327.73 329.05 92.89 x 309.95
	yolo11 resoing was successful ,and generated result.jpg
	
	real    0m1.018s
	user    0m2.956s
	sys     0m0.229s

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_416f16 416.jpeg
	3 = 0.96625 at 308.40 97.24 106.60 x 316.85
	3 = 0.96575 at 217.43 97.01 107.14 x 311.94
	3 = 0.95736 at 18.85 103.56 87.90 x 311.44
	3 = 0.94991 at 103.64 105.02 92.95 x 309.98
	3 = 0.87419 at 1.33 102.49 21.49 x 306.68
	yolo11 resoing was successful ,and generated result.jpg
	
	real    0m0.520s
	user    0m1.534s
	sys     0m0.170s

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_320f16 320.jpeg
	3 = 0.96755 at 211.97 0.25 107.03 x 318.15
	3 = 0.96709 at 121.75 0.47 107.09 x 311.34
	3 = 0.95825 at 3.00 3.72 98.06 x 314.93
	yolo11 resoing was successful ,and generated result.jpg
	
	real    0m0.416s
	user    0m1.244s
	sys     0m0.089s


三、Model conversion to INT8 format (It is strongly recommended to follow the comments and operation steps in the official yolo11.cpp source file of NCNN. Otherwise, it will not be possible to directly match and apply the official source file!!!).If you use the one-click conversion tool provided by the Ultralytics website to convert the file format to NCNN, you will need to modify the official yolo11.cpp source code of NCNN yourself. Moreover, this method cannot achieve the optimization benefits of NCNN.You can directly operate by opening the commented content in the official yolo11.cpp file: https://github.com/Tencent/ncnn/blob/master/examples/yolo11.cpp
！！！
At the same time, it is also necessary to refer to the operation steps of the quantization INT8 tool provided by the ncnn website:https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
！！！
This step involves first converting the .pt type file back to the FP32 type .param and .bin files, and then converting them again from the FP32 type .param and .bin files to the int8 type .param and .bin files.
---------------------------------------
1、Set up the environment(PC Anaconda Prompt terminal operation)
---------------------------------------
	conda activate (Environmental Name)
	pip3 install -U ultralytics pnnx ncnn

2、export yolo11 torchscript(PC Anaconda Prompt terminal operation)
---------------------------------------
Since my dataset is not from the official source, but rather a dataset that I trained myself to recognize 8 types of quantities.
In the Anaconda Prompt terminal of the PC, use the command `cd` to enter the folder where you have customized the model file named `best.pt`, and rename `best.pt` to `yolo11n.pt`.
	
	yolo export model=yolo11n.pt format=torchscript
3、convert torchscript with static shape(PC Anaconda Prompt terminal operation)
---------------------------------------
	pnnx yolo11n.torchscript fp16=0
	
4、modify yolo11n_pnnx.py for dynamic shape inference(PC Anaconda Prompt terminal operation)
---------------------------------------
	The above operation is handled in the same way....
	
5、re-export yolo11 torchscript(PC Anaconda Prompt terminal operation)
---------------------------------------
	The above operation is handled in the same way....
	
6、convert new torchscript with dynamic shape(PC Anaconda Prompt terminal operation)
---------------------------------------
	pnnx yolo11n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320] fp16=0

7、Optimize model(The official "quantized-int8-inference.md" can be skipped.)
---------------------------------------
NOTE: If your model is converted via pnnx, skip this step.

	./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0

8、Create the calibration table file(You can choose from the various methods provided in the official quantized-int8-inference.md document.)
---------------------------------------
I chose from the images. The official recommendation is to use the validation dataset for calibration. I transferred about 300 images from the data validation set stored on my PC to the Ubuntu virtual machine.
And create an "Images" folder in the "build-x86" folder to store it.
	//It is recommended to use relative paths.
	
	find $(pwd)/images -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) > imagelist.txt
	tools/quantize/ncnn2table yolo11n_pnnx.py.ncnn.param yolo11n_pnnx.py.ncnn.bin imagelist.txt yolo11n.table mean=[0,0,0] norm=[0.00392157,0.00392157,0.00392157] shape=[640,640,3] pixel=RGB thread=3 method=kl

Explanation of the operation steps for quantizing INT8 on the NCNN website：

*mean and norm are the values you passed to Mat::substract_mean_normalize()

*shape is the blob shape of your model, [w,h] or [w,h,c]

	* if w and h both are given, image will be resized to exactly size.
	
	* if w and h both are zero or negative, image will not be resized.
	
	* if only h is zero or negative, image's width will scaled resize to w, keeping aspect ratio.
	
	* if only w is zero or negative, image's height will scaled resize to h
	
*pixel is the pixel format of your model, image pixels will be converted to this type before Extractor::input()

*thread is the CPU thread count that could be used for parallel inference

*method is the post training quantization algorithm, kl and aciq are currently supported

The parameter values of "mean" and "norm" in the instruction are provided by AI. The "shape" parameter is aligned according to the "imgsz=640" setting I used when training the YOLO11 model. The "pixel" parameter is also recommended by AI for setting.

9、Quantize model(There are more settings to refer to in the official document "quantized-int8-inference.md".)
---------------------------------------
	tools/quantize/ncnn2int8 yolo11n_pnnx.py.ncnn.param yolo11n_pnnx.py.ncnn.bin yolo11n-int8.param yolo11n-int8.bin yolo11n.table

10、Modify the official yolo11.cpp source file in the Ubuntu virtual machine
---------------------------------------
By opening ncnn/
It is possible to manually disable the use of Vulkan acceleration. According to the AI query of the Zhi H616 chip, it is known that it does not support this feature. Additionally, I have personally tested it and found that it failed.
Here is the discussion about this issue: https://github.com/Tencent/ncnn/issues/6684
	
	yolo11.opt.use_vulkan_compute = false;
Modify the target size, as dynamic adjustment of the image input size has been set above. The smaller the target size, the less computational effort is required.

	const int target_size = (640 or 416 or 320);
Modify the type name of the identification. It can be updated by matching and modifying according to the "names" in the yaml file of the dataset.

	static const char* class_names[] = {}

Modify the names of imported files within the document. You can also modify the file names individually.

	yolo11.load_param("yolo11n-int8.param");
	yolo11.load_model("yolo11n-int8.bin");

11、Conduct tests on the development board
---------------------------------------
Make sure that the .param and .bin files transferred from the PC end and the yolo11 executable file transferred from the Ubuntu virtual machine are placed in the same directory on the development board.
At the same time, this path requires an image as an input parameter for the yolo11 executable file.
The following is a simple multiple test observation of a single run using the "time" command. The input images are respectively 640x640, 416x416, and 320x320.
On average, how much time is required：

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_640int8 640.jpeg
	3 = 0.94687 at 0.29 323.84 83.07 x 304.48
	3 = 0.94109 at 530.45 319.66 108.55 x 319.34
	3 = 0.93830 at 441.34 319.57 106.65 x 314.16
	3 = 0.89132 at 327.18 319.40 93.91 x 319.60
	3 = 0.87661 at 243.89 327.34 88.38 x 311.66
	3 = 0.87560 at 77.62 324.23 87.24 x 314.22
	3 = 0.81591 at 159.37 325.32 88.37 x 313.19
	yolo11 resoing was successful ,and generated result.jpg

	real    0m0.758s
	user    0m2.111s
	sys     0m0.167s

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_416int8 416.jpeg
	3 = 0.94254 at 217.01 94.95 107.01 x 314.32
	3 = 0.93293 at 307.28 97.18 107.72 x 317.82
	3 = 0.89480 at 18.48 102.16 89.22 x 312.50
	3 = 0.87872 at 103.55 97.00 92.32 x 318.00
	3 = 0.26192 at 0.00 155.10 24.79 x 250.49
	yolo11 resoing was successful ,and generated result.jpg

	real    0m0.393s
	user    0m1.068s
	sys     0m0.053s

	orangepi@orangepizero2:~/Downloads/drink_demo$ time ./yolo11_320int8 320.jpeg
	3 = 0.93579 at 4.63 2.75 96.12 x 316.25
	3 = 0.93541 at 117.42 1.60 110.16 x 309.45
	3 = 0.92949 at 211.92 0.57 107.08 x 318.43
	yolo11 resoing was successful ,and generated result.jpg

	real    0m0.292s
	user    0m0.729s
	sys     0m0.078s
