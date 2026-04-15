@echo off
setlocal

set GLSLC=..\.claude\tools\glslangValidator.exe

echo Compiling shaders...

rem === Matrix-vector (single token generation) ===
%GLSLC% -V matvec_f16.comp -o matvec_f16.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: matvec_f16.comp & exit /b 1 )

%GLSLC% -V matvec_q8_0.comp -o matvec_q8_0.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: matvec_q8_0.comp & exit /b 1 )

rem === Matrix-matrix (batched prefill) ===
%GLSLC% -V matmul_f16.comp -o matmul_f16.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: matmul_f16.comp & exit /b 1 )

%GLSLC% -V matmul_q8_0.comp -o matmul_q8_0.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: matmul_q8_0.comp & exit /b 1 )

rem === RMSNorm ===
%GLSLC% -V rmsnorm.comp -o rmsnorm.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rmsnorm.comp & exit /b 1 )

%GLSLC% -V rmsnorm_copy.comp -o rmsnorm_copy.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rmsnorm_copy.comp & exit /b 1 )

%GLSLC% -V rmsnorm_batch.comp -o rmsnorm_batch.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rmsnorm_batch.comp & exit /b 1 )

%GLSLC% -V rmsnorm_copy_batch.comp -o rmsnorm_copy_batch.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rmsnorm_copy_batch.comp & exit /b 1 )

rem === QK-norm and RoPE ===
%GLSLC% -V qk_norm.comp -o qk_norm.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: qk_norm.comp & exit /b 1 )

%GLSLC% -V rope.comp -o rope.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rope.comp & exit /b 1 )

%GLSLC% -V rope_batch.comp -o rope_batch.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: rope_batch.comp & exit /b 1 )

rem === Attention (single token — multi-head fused) ===
%GLSLC% -V attn_scores_mh.comp -o attn_scores_mh.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: attn_scores_mh.comp & exit /b 1 )

%GLSLC% -V softmax_mh.comp -o softmax_mh.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: softmax_mh.comp & exit /b 1 )

%GLSLC% -V attn_value_mh.comp -o attn_value_mh.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: attn_value_mh.comp & exit /b 1 )

rem === Attention (batched prefill — causal) ===
%GLSLC% -V attn_scores_prefill.comp -o attn_scores_prefill.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: attn_scores_prefill.comp & exit /b 1 )

%GLSLC% -V softmax_prefill.comp -o softmax_prefill.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: softmax_prefill.comp & exit /b 1 )

%GLSLC% -V attn_value_prefill.comp -o attn_value_prefill.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: attn_value_prefill.comp & exit /b 1 )

rem === KV cache store ===
%GLSLC% -V kv_cache_store.comp -o kv_cache_store.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: kv_cache_store.comp & exit /b 1 )

%GLSLC% -V kv_cache_store_batch.comp -o kv_cache_store_batch.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: kv_cache_store_batch.comp & exit /b 1 )

rem === Embedding lookup ===
%GLSLC% -V embed_lookup_f16.comp -o embed_lookup_f16.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: embed_lookup_f16.comp & exit /b 1 )

%GLSLC% -V embed_lookup_q8.comp -o embed_lookup_q8.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: embed_lookup_q8.comp & exit /b 1 )

%GLSLC% -V embed_lookup_batch_f16.comp -o embed_lookup_batch_f16.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: embed_lookup_batch_f16.comp & exit /b 1 )

%GLSLC% -V embed_lookup_batch_q8.comp -o embed_lookup_batch_q8.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: embed_lookup_batch_q8.comp & exit /b 1 )

rem === TurboQuant (TQ3 KV cache compression) ===
%GLSLC% -V tq3_quantize.comp -o tq3_quantize.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: tq3_quantize.comp & exit /b 1 )

%GLSLC% -V tq3_dequantize.comp -o tq3_dequantize.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: tq3_dequantize.comp & exit /b 1 )

%GLSLC% -V tq3_kv_quantize.comp -o tq3_kv_quantize.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: tq3_kv_quantize.comp & exit /b 1 )

%GLSLC% -V kv_cache_store_batch_tq3.comp -o kv_cache_store_batch_tq3.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: kv_cache_store_batch_tq3.comp & exit /b 1 )

%GLSLC% -V tq3_kv_dequantize.comp -o tq3_kv_dequantize.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: tq3_kv_dequantize.comp & exit /b 1 )

rem === Activation and residual ===
%GLSLC% -V gelu_mul.comp -o gelu_mul.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: gelu_mul.comp & exit /b 1 )

%GLSLC% -V vec_add.comp -o vec_add.spv
if %ERRORLEVEL% NEQ 0 ( echo FAILED: vec_add.comp & exit /b 1 )

echo Shaders compiled.

rem === Compile resource file ===
echo Compiling shader resources...
set BRCC32="%ProgramFiles(x86)%\Embarcadero\Studio\23.0\bin\brcc32.exe"
%BRCC32% VindexLLM.Shaders.rc -fo..\src\VindexLLM.Shaders.res
if %ERRORLEVEL% NEQ 0 ( echo FAILED: Resource compilation & exit /b 1 )

echo Done.
