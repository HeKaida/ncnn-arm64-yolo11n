# ncnn-arm64-yolo11n
The record shows that through the Ubuntu 22.04LTS virtual machine, the code was cross-compiled and deployed on the ARM64 architecture Orangepi Zero 2 1G development board for pure CPU inference in FP16 or INT8 format and image recognition.

***************************************
ubuntu虚拟机版本：
hkd@hkd-virtual-machine:~/Downloads/ncnn/build-cpu$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.5 LTS
Release:        22.04
Codename:       jammy
***************************************

***************************************
Orangepi Zero 2版本：
orangepi@orangepizero2:~/Downloads/drink_demo$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.2 LTS
Release:        22.04
Codename:       jammy
***************************************

***************************************
ubuntu虚拟机环境搭建(根据Orangepi Zero 2官方用户手册进行搭建)
***************************************
1、下载源码：
---------------------------------------
gitclonehttps://github.com/Tencent/ncnn.git

2、安装依赖包：
---------------------------------------
sudo apt update
sudo apt install build-essentialgitcmake libprotobuf-dev protobuf-compiler libopencv-dev

3、注释屏蔽benchncnn相关模型测试：
---------------------------------------
看个人喜好进行是否屏蔽，由于官方手册下载的ncnn不是最新的版本，官方只注释屏蔽vgg16这个模型测
---------------------------------------
//benchmark("alexnet", ncnn::Mat(227, 227, 3), opt, alexnet_param_data);
//benchmark("vgg16", ncnn::Mat(224, 224, 3), opt, vgg16_param_data);
//benchmark("vgg16_int8", ncnn::Mat(224, 224, 3), opt, vgg16_int8_param_data);
//benchmark("resnet50", ncnn::Mat(224, 224, 3), opt, resnet50_param_data);
//benchmark("resnet50_int8", ncnn::Mat(224, 224, 3), opt, resnet50_int8_param_data);

4、编译环境：
mkdir build
cd build
---------------------------------------
	在cmake之前请确保是否用有orangepi zero 2交叉编译工具，
	我使用的是gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu，
	并在~/.bashrc中添加环境变量指令(需要根据自己的路径修改)：
	export PATH=/home/$(whoami)/Downloads/orangepi-build-next/toolchains/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin:$PATH
----------------------------------------
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_EXAMPLES=ON
make -j$(nproc)

5、测试神经网络的推理性能：
---------------------------------------
详细操作可执行查看Orangepi Zero 2官方用户手册
通过SCP指令将ubuntu虚拟机上整个benchmark传输到开发板上进行测试
---------------------------------------

6、建议在ncnn文件夹目录下再创建一个build-x86(名字可自定义)文件夹
---------------------------------------
由于后期打算把模型转为int8类型的.param和.bin文件，
需要在PC或者ubuntu虚拟机使用ncnn2table和ncnn2int8可执行文件，
而前面创建的build文件夹是专门用于arm64架构的，
并没有加入-DNCNN_BUILD_TOOLS参数，导致在build文件夹下没有生成tools文件夹，
而且也不太可能在开发板上进行编译,
也不需要添加指定编译工具默认就是x86编译工具
---------------------------------------
mkdir build-x86
cd build-x86
cmake -DNCNN_SIMPLEOCV=ON -DNCNN_BUILD_TOOLS=ON
make -j$(nproc)
---------------------------------------
搭建完成后可以进入/build-x86/tools/quantize中查看是否又ncnn2table和ncnn2int8可执行文件，
若没有可以手动编译即可
---------------------------------------

***************************************
模型转换FP16格式(强烈推荐使用NCNN官方yolo11.cpp源文件里面的注释操作步骤，不然无法直接匹配并套用官方的源文件！！！)
若使用ultralytics官网的一键转ncnn文件格式工具，则需要你去自主修改NCNN官方yolo11.cpp源文件代码，同时也无法发挥到ncnn的优化！
可直接打开官方yolo11.cpp文件中注释内容进行操作：https://github.com/Tencent/ncnn/blob/master/examples/yolo11.cpp
默认是生成FP16格式大小的.param和.bin文件
***************************************
1、搭建环境(PC Anaconda Prompt终端操作)
---------------------------------------
conda activate (环境名称)
pip3 install -U ultralytics pnnx ncnn

2、export yolo11 torchscript(PC Anaconda Prompt终端操作)
---------------------------------------
由于我的数据集并不是官网的数据集，而是我自己训练的支持识别8个类型数量的数据集
在PC的Anaconda Prompt终端通过cd进入自己自定义存放best.pt的模型文件夹，并把best.pt重命名为yolo11n.pt
---------------------------------------
yolo export model=yolo11n.pt format=torchscript

3、convert torchscript with static shape(PC Anaconda Prompt终端操作)
---------------------------------------
pnnx yolo11n.torchscript

4、modify yolo11n_pnnx.py for dynamic shape inference(PC Anaconda Prompt终端操作)
---------------------------------------
//			A. modify reshape to support dynamic image sizes
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
对应自己的yolo11n_pnnx.py的内容进行修改，下面是我针对自己的生成的yolo11n_pnnx.py进行修改，
通过AI辅助查询才发现官方的yolo11n_pnnx.py里面的处理方式和我PC生成yolo11n_pnnx.py方式不太一样，
而且官方在这部分实例操作进行了简化，其实是需要把修改后的v_238 = torch.cat((v_235, v_236, v_237), dim=1)代码，
把下面所有代码直接删掉或者注释，然后在添加return v_238代码。
但是最终的效果都是一致，像我这种初学者不太懂得都是通过AI查询来解决的，如果有问题，可通过找AI解决
---------------------------------------
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
        # === 以下为修改部分 ===
        # 将 64维Box特征 (v_208) 和 8维Cls特征 (v_239) 拼接为 [1, 72, 8400]
        v_raw = torch.cat((v_208, v_239), dim=1)
        # 转置为 [1, 8400, 72]
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
---------------------------------------
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

5、re-export yolo11 torchscript(PC Anaconda Prompt终端操作)
---------------------------------------
python -c "import yolo11n_pnnx; yolo11n_pnnx.export_torchscript()"

6、convert new torchscript with dynamic shape(PC Anaconda Prompt终端操作)
---------------------------------------
pnnx yolo11n_pnnx.py.pt inputshape=[1,3,640,640] inputshape2=[1,3,320,320]

7、now you get ncnn model files(PC Anaconda Prompt终端操作)
---------------------------------------
mv yolo11n_pnnx.py.ncnn.param yolo11n.ncnn.param
mv yolo11n_pnnx.py.ncnn.bin yolo11n.ncnn.bin

8、在ubuntu虚拟机中修改官方yolo11.cpp源文件(若不是使用YOLO官方的数据集)
---------------------------------------
通过打开ncnn/
可以手动关闭使用vulkan加速，通过AI查询全志H616芯片得知不支持，同时我自己测试了是失败的，
这里有关这个问题的讨论：https://github.com/Tencent/ncnn/issues/6684，
---------------------------------------
yolo11.opt.use_vulkan_compute = false;
---------------------------------------
修改目标尺寸大小，因为上面设置了动态调整图像输入大小，目标尺寸越小，计算量就越少
---------------------------------------
const int target_size = (640或416或320);
---------------------------------------
修改识别类型名称，可以根据数据集的yaml文件的names进行匹配修改
---------------------------------------
static const char* class_names[] = {}

9、编译YOLO11.cpp源文件
---------------------------------------
在ncnn/build路径下make编译，即可在examples文件夹中自动覆盖为最新的yolo11可执行文件
---------------------------------------
make yolo11

10、部署开发板上测试
---------------------------------------
确保PC端传输.param和.bin与ubuntu虚拟机传输的yolo11可执行文件到开发板自定义同一路径下，
同时该路径需要一张图像作为yolo11可执行文件参数作为输入
已下是通过time指令来简单多次测试观察单次运行，分别以640*640、416*416、320*320的图像输入
平均需要多少时间，
---------------------------------------
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

***************************************
模型转换INT8格式(强烈推荐使用NCNN官方yolo11.cpp源文件里面的注释操作步骤，不然无法直接匹配并套用官方的源文件！！！)
若使用ultralytics官网的一键转ncnn文件格式工具，则需要你去自主修改NCNN官方yolo11.cpp源文件代码，同时也无法发挥到ncnn的优化！
可直接打开官方yolo11.cpp文件中注释内容进行操作：https://github.com/Tencent/ncnn/blob/master/examples/yolo11.cpp
！！！
同时还需要参考ncnn官网量化INT8的工具操作步骤：https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md
！！！
***************************************
