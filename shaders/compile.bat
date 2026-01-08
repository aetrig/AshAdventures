IF "%~1"=="" GOTO arg_fail
slangc shaders/%1.slang -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -entry fragMain -o shaders/%1.spv
GOTO end
:arg_fail
echo Supply slang file name
:end