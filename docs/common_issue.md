## Common issues

We list some common issues and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

<details>
<summary>
Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer";
</summary>

Please install the correct version of MMCV for the version of your MMDetection following the instruction above.

</details><br/>

<details>
<summary>
"No module named 'mmcv.ops'"; "No module named 'mmcv._ext'".
</summary>

1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.
2. Install mmcv-full following the instruction above.

</details><br/>

<details>
<summary>
"invalid device function" or "no kernel image is available for execution".
</summary>

1. Check the CUDA compute capability of you GPU.
2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture. You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV. The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
3. Check whether the running environment is the same as that when mmcv/mmdet is compiled. For example, you may compile mmcv using CUDA 10.0 bug run it on CUDA9.0 environments.

</details><br/>

<details>
<summary>
"undefined symbol" or "cannot open xxx.so"
</summary>

1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv.
2. If those symbols are Pytorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the Pytorch version is the same as that used for compiling mmcv.
3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built by and running on the same environment.

</details><br/>
