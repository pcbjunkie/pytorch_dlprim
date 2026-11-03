source_filename = "llvm-link"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1"
target triple = "r600-unknown-unknown"

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn
define dso_local amdgpu_kernel void @fill(i64 noundef %total, ptr addrspace(1) nocapture noundef writeonly align 4 %p, i64 noundef %p_offset, i64 noundef %seed, i64 noundef %seq, float noundef %v1, float noundef %v2) local_unnamed_addr #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = tail call i32 @llvm.r600.read.local.size.x() #3
  %1 = tail call i32 @llvm.r600.read.tidig.x(), !range !8
  %2 = tail call i32 @llvm.r600.read.tgid.x()
  %3 = mul i32 %0, %2
  %4 = add i32 %3, %1
  %5 = tail call ptr addrspace(7) @llvm.r600.implicitarg.ptr()
  %6 = getelementptr inbounds i32, ptr addrspace(7) %5, i32 1
  %7 = load i32, ptr addrspace(7) %6, align 4, !tbaa !9
  %8 = add i32 %4, %7
  %conv = zext i32 %8 to i64
  %mul = shl nuw nsw i64 %conv, 2
  %cmp.not = icmp ult i64 %mul, %total
  br i1 %cmp.not, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  %add = add i64 %conv, %seq
  %shr.i = lshr i64 %add, 32
  %conv1.i = trunc i64 %add to i32
  %conv2.i = trunc i64 %seed to i32
  %shr3.i = lshr i64 %seed, 32
  %conv4.i = trunc i64 %shr3.i to i32
  %mul.i.i.i.i = mul i32 %conv1.i, -766435501
  %xor9.i.i.i4 = xor i64 %shr.i, %shr3.i
  %xor9.i.i.i = trunc i64 %xor9.i.i.i4 to i32
  %xor11.i.i.i = xor i32 %mul.i.i.i.i, %xor9.i.i.i
  %add.i.i.i = add i32 %conv2.i, -1150833019
  %add16.i.i.i = add i32 %conv4.i, -1640531527
  %mul.i.i.1.i.i = mul i32 %conv2.i, -766435501
  %conv.i.i.1.i.i = zext i32 %xor11.i.i.i to i64
  %mul.i22.i.1.i.i = mul nuw i64 %conv.i.i.1.i.i, 3449720151
  %shr.i.i.1.i.i = lshr i64 %mul.i22.i.1.i.i, 32
  %conv2.i.i.1.i.i = trunc i64 %shr.i.i.1.i.i to i32
  %xor2.i.1.i.i = xor i32 %add.i.i.i, %conv2.i.i.1.i.i
  %mul.i23.i.1.i.i = mul i32 %xor11.i.i.i, -845247145
  %xor9.i.1.i.i = xor i32 %add16.i.i.i, %mul.i.i.1.i.i
  %xor11.i.1.i.i = xor i32 %xor9.i.1.i.i, %mul.i.i.i.i
  %add.i.1.i.i = add i32 %conv2.i, 1993301258
  %add16.i.1.i.i = add i32 %conv4.i, 1013904242
  %mul.i.i.2.i.i = mul i32 %xor2.i.1.i.i, -766435501
  %conv.i.i.2.i.i = zext i32 %xor11.i.1.i.i to i64
  %mul.i22.i.2.i.i = mul nuw i64 %conv.i.i.2.i.i, 3449720151
  %shr.i.i.2.i.i = lshr i64 %mul.i22.i.2.i.i, 32
  %conv2.i.i.2.i.i = trunc i64 %shr.i.i.2.i.i to i32
  %xor.i.2.i.i = xor i32 %mul.i23.i.1.i.i, %add.i.1.i.i
  %xor2.i.2.i.i = xor i32 %xor.i.2.i.i, %conv2.i.i.2.i.i
  %mul.i23.i.2.i.i = mul i32 %xor11.i.1.i.i, -845247145
  %xor9.i.2.i.i = xor i32 %add16.i.1.i.i, %mul.i.i.1.i.i
  %xor11.i.2.i.i = xor i32 %xor9.i.2.i.i, %mul.i.i.2.i.i
  %add.i.2.i.i = add i32 %conv2.i, 842468239
  %add16.i.2.i.i = add i32 %conv4.i, -626627285
  %mul.i.i.3.i.i = mul i32 %xor2.i.2.i.i, -766435501
  %conv.i.i.3.i.i = zext i32 %xor11.i.2.i.i to i64
  %mul.i22.i.3.i.i = mul nuw i64 %conv.i.i.3.i.i, 3449720151
  %shr.i.i.3.i.i = lshr i64 %mul.i22.i.3.i.i, 32
  %conv2.i.i.3.i.i = trunc i64 %shr.i.i.3.i.i to i32
  %xor.i.3.i.i = xor i32 %mul.i23.i.2.i.i, %add.i.2.i.i
  %xor2.i.3.i.i = xor i32 %xor.i.3.i.i, %conv2.i.i.3.i.i
  %mul.i23.i.3.i.i = mul i32 %xor11.i.2.i.i, -845247145
  %xor9.i.3.i.i = xor i32 %mul.i.i.2.i.i, %add16.i.2.i.i
  %xor11.i.3.i.i = xor i32 %xor9.i.3.i.i, %mul.i.i.3.i.i
  %add.i.3.i.i = add i32 %conv2.i, -308364780
  %add16.i.3.i.i = add i32 %conv4.i, 2027808484
  %mul.i.i.4.i.i = mul i32 %xor2.i.3.i.i, -766435501
  %conv.i.i.4.i.i = zext i32 %xor11.i.3.i.i to i64
  %mul.i22.i.4.i.i = mul nuw i64 %conv.i.i.4.i.i, 3449720151
  %shr.i.i.4.i.i = lshr i64 %mul.i22.i.4.i.i, 32
  %conv2.i.i.4.i.i = trunc i64 %shr.i.i.4.i.i to i32
  %xor.i.4.i.i = xor i32 %mul.i23.i.3.i.i, %add.i.3.i.i
  %xor2.i.4.i.i = xor i32 %xor.i.4.i.i, %conv2.i.i.4.i.i
  %mul.i23.i.4.i.i = mul i32 %xor11.i.3.i.i, -845247145
  %xor9.i.4.i.i = xor i32 %mul.i.i.3.i.i, %add16.i.3.i.i
  %xor11.i.4.i.i = xor i32 %xor9.i.4.i.i, %mul.i.i.4.i.i
  %add.i.4.i.i = add i32 %conv2.i, -1459197799
  %add16.i.4.i.i = add i32 %conv4.i, 387276957
  %mul.i.i.5.i.i = mul i32 %xor2.i.4.i.i, -766435501
  %conv.i.i.5.i.i = zext i32 %xor11.i.4.i.i to i64
  %mul.i22.i.5.i.i = mul nuw i64 %conv.i.i.5.i.i, 3449720151
  %shr.i.i.5.i.i = lshr i64 %mul.i22.i.5.i.i, 32
  %conv2.i.i.5.i.i = trunc i64 %shr.i.i.5.i.i to i32
  %xor.i.5.i.i = xor i32 %mul.i23.i.4.i.i, %add.i.4.i.i
  %xor2.i.5.i.i = xor i32 %xor.i.5.i.i, %conv2.i.i.5.i.i
  %mul.i23.i.5.i.i = mul i32 %xor11.i.4.i.i, -845247145
  %xor9.i.5.i.i = xor i32 %mul.i.i.4.i.i, %add16.i.4.i.i
  %xor11.i.5.i.i = xor i32 %xor9.i.5.i.i, %mul.i.i.5.i.i
  %add.i.5.i.i = add i32 %conv2.i, 1684936478
  %add16.i.5.i.i = add i32 %conv4.i, -1253254570
  %mul.i.i.6.i.i = mul i32 %xor2.i.5.i.i, -766435501
  %conv.i.i.6.i.i = zext i32 %xor11.i.5.i.i to i64
  %mul.i22.i.6.i.i = mul nuw i64 %conv.i.i.6.i.i, 3449720151
  %shr.i.i.6.i.i = lshr i64 %mul.i22.i.6.i.i, 32
  %conv2.i.i.6.i.i = trunc i64 %shr.i.i.6.i.i to i32
  %xor.i.6.i.i = xor i32 %mul.i23.i.5.i.i, %add.i.5.i.i
  %xor2.i.6.i.i = xor i32 %xor.i.6.i.i, %conv2.i.i.6.i.i
  %mul.i23.i.6.i.i = mul i32 %xor11.i.5.i.i, -845247145
  %xor9.i.6.i.i = xor i32 %mul.i.i.5.i.i, %add16.i.5.i.i
  %xor11.i.6.i.i = xor i32 %xor9.i.6.i.i, %mul.i.i.6.i.i
  %add.i.6.i.i = add i32 %conv2.i, 534103459
  %add16.i.6.i.i = add i32 %conv4.i, 1401181199
  %mul.i.i.7.i.i = mul i32 %xor2.i.6.i.i, -766435501
  %conv.i.i.7.i.i = zext i32 %xor11.i.6.i.i to i64
  %mul.i22.i.7.i.i = mul nuw i64 %conv.i.i.7.i.i, 3449720151
  %shr.i.i.7.i.i = lshr i64 %mul.i22.i.7.i.i, 32
  %conv2.i.i.7.i.i = trunc i64 %shr.i.i.7.i.i to i32
  %xor.i.7.i.i = xor i32 %mul.i23.i.6.i.i, %add.i.6.i.i
  %xor2.i.7.i.i = xor i32 %xor.i.7.i.i, %conv2.i.i.7.i.i
  %mul.i23.i.7.i.i = mul i32 %xor11.i.6.i.i, -845247145
  %xor9.i.7.i.i = xor i32 %mul.i.i.6.i.i, %add16.i.6.i.i
  %xor11.i.7.i.i = xor i32 %xor9.i.7.i.i, %mul.i.i.7.i.i
  %add.i.7.i.i = add i32 %conv2.i, -616729560
  %add16.i.7.i.i = add i32 %conv4.i, -239350328
  %mul.i.i.8.i.i = mul i32 %xor2.i.7.i.i, -766435501
  %conv.i.i.8.i.i = zext i32 %xor11.i.7.i.i to i64
  %mul.i22.i.8.i.i = mul nuw i64 %conv.i.i.8.i.i, 3449720151
  %shr.i.i.8.i.i = lshr i64 %mul.i22.i.8.i.i, 32
  %conv2.i.i.8.i.i = trunc i64 %shr.i.i.8.i.i to i32
  %xor.i.8.i.i = xor i32 %mul.i23.i.7.i.i, %add.i.7.i.i
  %xor2.i.8.i.i = xor i32 %xor.i.8.i.i, %conv2.i.i.8.i.i
  %mul.i23.i.8.i.i = mul i32 %xor11.i.7.i.i, -845247145
  %xor9.i.8.i.i = xor i32 %mul.i.i.7.i.i, %add16.i.7.i.i
  %xor11.i.8.i.i = xor i32 %xor9.i.8.i.i, %mul.i.i.8.i.i
  %add.i.8.i.i = add i32 %conv2.i, -1767562579
  %add16.i.8.i.i = add i32 %conv4.i, -1879881855
  %conv.i.i.9.i.i = zext i32 %xor11.i.8.i.i to i64
  %mul.i22.i.9.i.i = mul nuw i64 %conv.i.i.9.i.i, 3449720151
  %shr.i.i.9.i.i = lshr i64 %mul.i22.i.9.i.i, 32
  %conv2.i.i.9.i.i = trunc i64 %shr.i.i.9.i.i to i32
  %xor.i.9.i.i = xor i32 %mul.i23.i.8.i.i, %add.i.8.i.i
  %xor9.i.9.i.i = xor i32 %mul.i.i.8.i.i, %add16.i.8.i.i
  %mul.i.i.9.i.i = mul i32 %xor2.i.8.i.i, -766435501
  %xor2.i.9.i.i = xor i32 %xor.i.9.i.i, %conv2.i.i.9.i.i
  %mul.i23.i.9.i.i = mul i32 %xor11.i.8.i.i, -845247145
  %xor11.i.9.i.i = xor i32 %xor9.i.9.i.i, %mul.i.i.9.i.i
  %9 = insertelement <4 x i32> poison, i32 %mul.i23.i.9.i.i, i64 0
  %10 = insertelement <4 x i32> %9, i32 %xor11.i.9.i.i, i64 1
  %11 = insertelement <4 x i32> %10, i32 %mul.i.i.9.i.i, i64 2
  %12 = insertelement <4 x i32> %11, i32 %xor2.i.9.i.i, i64 3
  %13 = lshr <4 x i32> %12, <i32 8, i32 8, i32 8, i32 8>
  %14 = uitofp <4 x i32> %13 to <4 x float>
  %15 = fmul <4 x float> %14, <float 0x3E70000000000000, float 0x3E70000000000000, float 0x3E70000000000000, float 0x3E70000000000000>
  %sub = fsub float %v2, %v1
  %splat.splatinsert = insertelement <4 x float> poison, float %sub, i64 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %splat.splatinsert5 = insertelement <4 x float> poison, float %v1, i64 0
  %splat.splat6 = shufflevector <4 x float> %splat.splatinsert5, <4 x float> poison, <4 x i32> zeroinitializer
  %16 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %15, <4 x float> %splat.splat, <4 x float> %splat.splat6)
  %idx.ext = trunc i64 %p_offset to i32
  %add.ptr = getelementptr inbounds float, ptr addrspace(1) %p, i32 %idx.ext
  %idx.ext11 = trunc i64 %mul to i32
  %add.ptr12 = getelementptr inbounds float, ptr addrspace(1) %add.ptr, i32 %idx.ext11
  store <4 x float> %16, ptr addrspace(1) %add.ptr12, align 4, !tbaa !13
  br label %cleanup

cleanup:                                          ; preds = %if.end, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.r600.read.tgid.x() #2

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.r600.read.local.size.x() #2

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.r600.read.tidig.x() #2

; Function Attrs: nounwind readnone speculatable willreturn
declare ptr addrspace(7) @llvm.r600.implicitarg.ptr() #2

attributes #0 = { convergent mustprogress nofree norecurse nosync nounwind willreturn "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="none" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cayman" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { convergent nobuiltin "no-builtins" }

!opencl.ocl.version = !{!0}
!llvm.module.flags = !{!1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"Debian clang version 15.0.6"}
!4 = !{i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!6 = !{!"ulong", !"float*", !"ulong", !"ulong", !"ulong", !"float", !"float"}
!7 = !{!"", !"", !"", !"", !"", !"", !""}
!8 = !{i32 0, i32 1024}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}
!13 = !{!11, !11, i64 0}
