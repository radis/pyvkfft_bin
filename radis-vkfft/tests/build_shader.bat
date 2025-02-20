:: TODO: The --target-env=vulkan1.0 should maybe be left out (but it's working now so I don't want to touch it)
:: glslc -O --target-env=vulkan1.0 -ocmdApplyTestLineshape.spv cmdApplyTestLineshape.comp
glslc -O --target-env=vulkan1.0 -ocmdTestShader1.spv cmdTestShader1.comp
glslc -O --target-env=vulkan1.0 -ocmdTestFillLDM.spv cmdTestFillLDM.comp
glslc -O --target-env=vulkan1.0 -ocmdTestApplyLineshapes.spv cmdTestApplyLineshapes.comp
glslc -O --target-env=vulkan1.0 -oout.spv FFT1_kernel_3.comp
pause
