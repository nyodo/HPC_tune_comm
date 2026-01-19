; ModuleID = '/thfs3/home/xjtu_cx/AMT-Toolkits/tests/mem/FCJM-TEST/auto-test/MT/2DCONV/2DConvolution.dev.ll'
source_filename = "/thfs3/home/xjtu_cx/AMT-Toolkits/tests/mem/FCJM-TEST/auto-test/MT/2DCONV/2DConvolution.dev.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = external hidden unnamed_addr constant [16 x i8], align 1
@kernel = external hidden global ptr, align 8
@.str.1 = external hidden unnamed_addr constant [11 x i8], align 1
@.str.2 = external hidden unnamed_addr constant [10 x i8], align 1
@.str.3 = external hidden unnamed_addr constant [17 x i8], align 1
@.str.4 = external hidden unnamed_addr constant [4 x i8], align 1
@.str.5 = external hidden unnamed_addr constant [2 x i8], align 1
@.str.6 = external hidden unnamed_addr constant [6 x i8], align 1
@.str.7 = external hidden unnamed_addr constant [16 x i8], align 1
@.str.8 = external hidden unnamed_addr constant [12 x i8], align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @convolution2D_kernel(i32 noundef %ni, i32 noundef %nj, ptr noundef %A, ptr noundef %B) #0 section ".global" {
entry:
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %A.addr = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  %group_size = alloca i32, align 4
  %thread_id = alloca i32, align 4
  %eid = alloca i32, align 4
  %before_hot_data = alloca [27 x i64], align 8
  %after_hot_data = alloca [27 x i64], align 8
  %c11 = alloca float, align 4
  %c12 = alloca float, align 4
  %c13 = alloca float, align 4
  %c21 = alloca float, align 4
  %c22 = alloca float, align 4
  %c23 = alloca float, align 4
  %c31 = alloca float, align 4
  %c32 = alloca float, align 4
  %c33 = alloca float, align 4
  %total_tasks = alloca i32, align 4
  %base_tasks = alloca i32, align 4
  %remainder = alloca i32, align 4
  %start = alloca i32, align 4
  %end = alloca i32, align 4
  %t = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %eid116 = alloca i32, align 4
  %eid129 = alloca i32, align 4
  %eid142 = alloca i32, align 4
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  store ptr %A, ptr %A.addr, align 8
  store ptr %B, ptr %B.addr, align 8
  %call = call i32 @get_group_size()
  store i32 %call, ptr %group_size, align 4
  %call1 = call i32 @get_thread_id()
  store i32 %call1, ptr %thread_id, align 4
  %0 = load i32, ptr %thread_id, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load ptr, ptr @kernel, align 8
  call void (ptr, ...) @hthread_printf(ptr noundef @.str, ptr noundef %1)
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.1, i32 noundef 1024)
  %2 = load i32, ptr %thread_id, align 4
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.2, i32 noundef %2)
  store i32 0, ptr %eid, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %3 = load i32, ptr %eid, align 4
  %cmp2 = icmp slt i32 %3, 26
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, ptr %eid, align 4
  call void @prof_start(i32 noundef %4)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %5 = load i32, ptr %eid, align 4
  %inc = add nsw i32 %5, 1
  store i32 %inc, ptr %eid, align 4
  br label %for.cond, !llvm.loop !6

for.end:                                          ; preds = %for.cond
  store i32 0, ptr %eid, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc7, %for.end
  %6 = load i32, ptr %eid, align 4
  %cmp4 = icmp slt i32 %6, 26
  br i1 %cmp4, label %for.body5, label %for.end9

for.body5:                                        ; preds = %for.cond3
  %7 = load i32, ptr %eid, align 4
  %call6 = call i64 @prof_read(i32 noundef %7)
  %8 = load i32, ptr %eid, align 4
  %idxprom = sext i32 %8 to i64
  %arrayidx = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 %idxprom
  store i64 %call6, ptr %arrayidx, align 8
  br label %for.inc7

for.inc7:                                         ; preds = %for.body5
  %9 = load i32, ptr %eid, align 4
  %inc8 = add nsw i32 %9, 1
  store i32 %inc8, ptr %eid, align 4
  br label %for.cond3, !llvm.loop !8

for.end9:                                         ; preds = %for.cond3
  %call10 = call i64 @get_clk()
  %arrayidx11 = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 26
  store i64 %call10, ptr %arrayidx11, align 8
  br label %if.end

if.end:                                           ; preds = %for.end9, %entry
  store float 0x3FC99999A0000000, ptr %c11, align 4
  store float 5.000000e-01, ptr %c21, align 4
  store float 0xBFE99999A0000000, ptr %c31, align 4
  store float 0xBFD3333340000000, ptr %c12, align 4
  store float 0x3FE3333340000000, ptr %c22, align 4
  store float 0xBFECCCCCC0000000, ptr %c32, align 4
  store float 0x3FD99999A0000000, ptr %c13, align 4
  store float 0x3FE6666660000000, ptr %c23, align 4
  store float 0x3FB99999A0000000, ptr %c33, align 4
  %10 = load i32, ptr %ni.addr, align 4
  %sub = sub nsw i32 %10, 2
  %11 = load i32, ptr %nj.addr, align 4
  %sub12 = sub nsw i32 %11, 2
  %mul = mul nsw i32 %sub, %sub12
  store i32 %mul, ptr %total_tasks, align 4
  %12 = load i32, ptr %total_tasks, align 4
  %cmp13 = icmp sle i32 %12, 0
  br i1 %cmp13, label %if.then14, label %if.end15

if.then14:                                        ; preds = %if.end
  br label %if.end160

if.end15:                                         ; preds = %if.end
  %13 = load i32, ptr %total_tasks, align 4
  %14 = load i32, ptr %group_size, align 4
  %div = sdiv i32 %13, %14
  store i32 %div, ptr %base_tasks, align 4
  %15 = load i32, ptr %total_tasks, align 4
  %16 = load i32, ptr %group_size, align 4
  %rem = srem i32 %15, %16
  store i32 %rem, ptr %remainder, align 4
  %17 = load i32, ptr %thread_id, align 4
  %18 = load i32, ptr %remainder, align 4
  %cmp16 = icmp slt i32 %17, %18
  br i1 %cmp16, label %cond.true, label %cond.false

cond.true:                                        ; preds = %if.end15
  %19 = load i32, ptr %thread_id, align 4
  %20 = load i32, ptr %base_tasks, align 4
  %add = add nsw i32 %20, 1
  %mul17 = mul nsw i32 %19, %add
  br label %cond.end

cond.false:                                       ; preds = %if.end15
  %21 = load i32, ptr %remainder, align 4
  %22 = load i32, ptr %base_tasks, align 4
  %add18 = add nsw i32 %22, 1
  %mul19 = mul nsw i32 %21, %add18
  %23 = load i32, ptr %thread_id, align 4
  %24 = load i32, ptr %remainder, align 4
  %sub20 = sub nsw i32 %23, %24
  %25 = load i32, ptr %base_tasks, align 4
  %mul21 = mul nsw i32 %sub20, %25
  %add22 = add nsw i32 %mul19, %mul21
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %mul17, %cond.true ], [ %add22, %cond.false ]
  store i32 %cond, ptr %start, align 4
  %26 = load i32, ptr %start, align 4
  %27 = load i32, ptr %thread_id, align 4
  %28 = load i32, ptr %remainder, align 4
  %cmp23 = icmp slt i32 %27, %28
  br i1 %cmp23, label %cond.true24, label %cond.false26

cond.true24:                                      ; preds = %cond.end
  %29 = load i32, ptr %base_tasks, align 4
  %add25 = add nsw i32 %29, 1
  br label %cond.end27

cond.false26:                                     ; preds = %cond.end
  %30 = load i32, ptr %base_tasks, align 4
  br label %cond.end27

cond.end27:                                       ; preds = %cond.false26, %cond.true24
  %cond28 = phi i32 [ %add25, %cond.true24 ], [ %30, %cond.false26 ]
  %add29 = add nsw i32 %26, %cond28
  store i32 %add29, ptr %end, align 4
  %31 = load i32, ptr %start, align 4
  store i32 %31, ptr %t, align 4
  br label %for.cond30

for.cond30:                                       ; preds = %for.inc100, %cond.end27
  %32 = load i32, ptr %t, align 4
  %33 = load i32, ptr %end, align 4
  %cmp31 = icmp slt i32 %32, %33
  br i1 %cmp31, label %for.body32, label %for.end102

for.body32:                                       ; preds = %for.cond30
  %34 = load i32, ptr %t, align 4
  %35 = load i32, ptr %nj.addr, align 4
  %sub33 = sub nsw i32 %35, 2
  %div34 = sdiv i32 %34, %sub33
  %add35 = add nsw i32 1, %div34
  store i32 %add35, ptr %i, align 4
  %36 = load i32, ptr %t, align 4
  %37 = load i32, ptr %nj.addr, align 4
  %sub36 = sub nsw i32 %37, 2
  %rem37 = srem i32 %36, %sub36
  %add38 = add nsw i32 1, %rem37
  store i32 %add38, ptr %j, align 4
  %38 = load float, ptr %c11, align 4
  %39 = load ptr, ptr %A.addr, align 8
  %40 = load i32, ptr %i, align 4
  %sub39 = sub nsw i32 %40, 1
  %41 = load i32, ptr %nj.addr, align 4
  %mul40 = mul nsw i32 %sub39, %41
  %42 = load i32, ptr %j, align 4
  %sub41 = sub nsw i32 %42, 1
  %add42 = add nsw i32 %mul40, %sub41
  %idxprom43 = sext i32 %add42 to i64
  %arrayidx44 = getelementptr inbounds float, ptr %39, i64 %idxprom43
  %43 = load float, ptr %arrayidx44, align 4
  %44 = load float, ptr %c21, align 4
  %45 = load ptr, ptr %A.addr, align 8
  %46 = load i32, ptr %i, align 4
  %sub46 = sub nsw i32 %46, 1
  %47 = load i32, ptr %nj.addr, align 4
  %mul47 = mul nsw i32 %sub46, %47
  %48 = load i32, ptr %j, align 4
  %add48 = add nsw i32 %mul47, %48
  %idxprom49 = sext i32 %add48 to i64
  %arrayidx50 = getelementptr inbounds float, ptr %45, i64 %idxprom49
  %49 = load float, ptr %arrayidx50, align 4
  %mul51 = fmul float %44, %49
  %50 = call float @llvm.fmuladd.f32(float %38, float %43, float %mul51)
  %51 = load float, ptr %c31, align 4
  %52 = load ptr, ptr %A.addr, align 8
  %53 = load i32, ptr %i, align 4
  %sub52 = sub nsw i32 %53, 1
  %54 = load i32, ptr %nj.addr, align 4
  %mul53 = mul nsw i32 %sub52, %54
  %55 = load i32, ptr %j, align 4
  %add54 = add nsw i32 %55, 1
  %add55 = add nsw i32 %mul53, %add54
  %idxprom56 = sext i32 %add55 to i64
  %arrayidx57 = getelementptr inbounds float, ptr %52, i64 %idxprom56
  %56 = load float, ptr %arrayidx57, align 4
  %57 = call float @llvm.fmuladd.f32(float %51, float %56, float %50)
  %58 = load float, ptr %c12, align 4
  %59 = load ptr, ptr %A.addr, align 8
  %60 = load i32, ptr %i, align 4
  %61 = load i32, ptr %nj.addr, align 4
  %mul59 = mul nsw i32 %60, %61
  %62 = load i32, ptr %j, align 4
  %sub60 = sub nsw i32 %62, 1
  %add61 = add nsw i32 %mul59, %sub60
  %idxprom62 = sext i32 %add61 to i64
  %arrayidx63 = getelementptr inbounds float, ptr %59, i64 %idxprom62
  %63 = load float, ptr %arrayidx63, align 4
  %64 = call float @llvm.fmuladd.f32(float %58, float %63, float %57)
  %65 = load float, ptr %c22, align 4
  %66 = load ptr, ptr %A.addr, align 8
  %67 = load i32, ptr %i, align 4
  %68 = load i32, ptr %nj.addr, align 4
  %mul65 = mul nsw i32 %67, %68
  %69 = load i32, ptr %j, align 4
  %add66 = add nsw i32 %mul65, %69
  %idxprom67 = sext i32 %add66 to i64
  %arrayidx68 = getelementptr inbounds float, ptr %66, i64 %idxprom67
  %70 = load float, ptr %arrayidx68, align 4
  %71 = call float @llvm.fmuladd.f32(float %65, float %70, float %64)
  %72 = load float, ptr %c32, align 4
  %73 = load ptr, ptr %A.addr, align 8
  %74 = load i32, ptr %i, align 4
  %75 = load i32, ptr %nj.addr, align 4
  %mul70 = mul nsw i32 %74, %75
  %76 = load i32, ptr %j, align 4
  %add71 = add nsw i32 %76, 1
  %add72 = add nsw i32 %mul70, %add71
  %idxprom73 = sext i32 %add72 to i64
  %arrayidx74 = getelementptr inbounds float, ptr %73, i64 %idxprom73
  %77 = load float, ptr %arrayidx74, align 4
  %78 = call float @llvm.fmuladd.f32(float %72, float %77, float %71)
  %79 = load float, ptr %c13, align 4
  %80 = load ptr, ptr %A.addr, align 8
  %81 = load i32, ptr %i, align 4
  %add76 = add nsw i32 %81, 1
  %82 = load i32, ptr %nj.addr, align 4
  %mul77 = mul nsw i32 %add76, %82
  %83 = load i32, ptr %j, align 4
  %sub78 = sub nsw i32 %83, 1
  %add79 = add nsw i32 %mul77, %sub78
  %idxprom80 = sext i32 %add79 to i64
  %arrayidx81 = getelementptr inbounds float, ptr %80, i64 %idxprom80
  %84 = load float, ptr %arrayidx81, align 4
  %85 = call float @llvm.fmuladd.f32(float %79, float %84, float %78)
  %86 = load float, ptr %c23, align 4
  %87 = load ptr, ptr %A.addr, align 8
  %88 = load i32, ptr %i, align 4
  %add83 = add nsw i32 %88, 1
  %89 = load i32, ptr %nj.addr, align 4
  %mul84 = mul nsw i32 %add83, %89
  %90 = load i32, ptr %j, align 4
  %add85 = add nsw i32 %mul84, %90
  %idxprom86 = sext i32 %add85 to i64
  %arrayidx87 = getelementptr inbounds float, ptr %87, i64 %idxprom86
  %91 = load float, ptr %arrayidx87, align 4
  %92 = call float @llvm.fmuladd.f32(float %86, float %91, float %85)
  %93 = load float, ptr %c33, align 4
  %94 = load ptr, ptr %A.addr, align 8
  %95 = load i32, ptr %i, align 4
  %add89 = add nsw i32 %95, 1
  %96 = load i32, ptr %nj.addr, align 4
  %mul90 = mul nsw i32 %add89, %96
  %97 = load i32, ptr %j, align 4
  %add91 = add nsw i32 %97, 1
  %add92 = add nsw i32 %mul90, %add91
  %idxprom93 = sext i32 %add92 to i64
  %arrayidx94 = getelementptr inbounds float, ptr %94, i64 %idxprom93
  %98 = load float, ptr %arrayidx94, align 4
  %99 = call float @llvm.fmuladd.f32(float %93, float %98, float %92)
  %100 = load ptr, ptr %B.addr, align 8
  %101 = load i32, ptr %i, align 4
  %102 = load i32, ptr %nj.addr, align 4
  %mul96 = mul nsw i32 %101, %102
  %103 = load i32, ptr %j, align 4
  %add97 = add nsw i32 %mul96, %103
  %idxprom98 = sext i32 %add97 to i64
  %arrayidx99 = getelementptr inbounds float, ptr %100, i64 %idxprom98
  store float %99, ptr %arrayidx99, align 4
  br label %for.inc100

for.inc100:                                       ; preds = %for.body32
  %104 = load i32, ptr %t, align 4
  %inc101 = add nsw i32 %104, 1
  store i32 %inc101, ptr %t, align 4
  br label %for.cond30, !llvm.loop !9

for.end102:                                       ; preds = %for.cond30
  %105 = load i32, ptr %thread_id, align 4
  %cmp103 = icmp eq i32 %105, 0
  br i1 %cmp103, label %if.then104, label %if.end160

if.then104:                                       ; preds = %for.end102
  %call105 = call i64 @get_clk()
  %arrayidx106 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 26
  store i64 %call105, ptr %arrayidx106, align 8
  store i32 0, ptr %eid, align 4
  br label %for.cond107

for.cond107:                                      ; preds = %for.inc113, %if.then104
  %106 = load i32, ptr %eid, align 4
  %cmp108 = icmp slt i32 %106, 26
  br i1 %cmp108, label %for.body109, label %for.end115

for.body109:                                      ; preds = %for.cond107
  %107 = load i32, ptr %eid, align 4
  %call110 = call i64 @prof_end(i32 noundef %107)
  %108 = load i32, ptr %eid, align 4
  %idxprom111 = sext i32 %108 to i64
  %arrayidx112 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 %idxprom111
  store i64 %call110, ptr %arrayidx112, align 8
  br label %for.inc113

for.inc113:                                       ; preds = %for.body109
  %109 = load i32, ptr %eid, align 4
  %inc114 = add nsw i32 %109, 1
  store i32 %inc114, ptr %eid, align 4
  br label %for.cond107, !llvm.loop !10

for.end115:                                       ; preds = %for.cond107
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.3)
  store i32 0, ptr %eid116, align 4
  br label %for.cond117

for.cond117:                                      ; preds = %for.inc125, %for.end115
  %110 = load i32, ptr %eid116, align 4
  %cmp118 = icmp slt i32 %110, 26
  br i1 %cmp118, label %for.body119, label %for.end127

for.body119:                                      ; preds = %for.cond117
  %111 = load i32, ptr %eid116, align 4
  %idxprom120 = sext i32 %111 to i64
  %arrayidx121 = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 %idxprom120
  %112 = load i64, ptr %arrayidx121, align 8
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.4, i64 noundef %112)
  %113 = load i32, ptr %eid116, align 4
  %cmp122 = icmp slt i32 %113, 25
  br i1 %cmp122, label %if.then123, label %if.end124

if.then123:                                       ; preds = %for.body119
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.5)
  br label %if.end124

if.end124:                                        ; preds = %if.then123, %for.body119
  br label %for.inc125

for.inc125:                                       ; preds = %if.end124
  %114 = load i32, ptr %eid116, align 4
  %inc126 = add nsw i32 %114, 1
  store i32 %inc126, ptr %eid116, align 4
  br label %for.cond117, !llvm.loop !11

for.end127:                                       ; preds = %for.cond117
  %arrayidx128 = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 26
  %115 = load i64, ptr %arrayidx128, align 8
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.6, i64 noundef %115)
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.7)
  store i32 0, ptr %eid129, align 4
  br label %for.cond130

for.cond130:                                      ; preds = %for.inc138, %for.end127
  %116 = load i32, ptr %eid129, align 4
  %cmp131 = icmp slt i32 %116, 26
  br i1 %cmp131, label %for.body132, label %for.end140

for.body132:                                      ; preds = %for.cond130
  %117 = load i32, ptr %eid129, align 4
  %idxprom133 = sext i32 %117 to i64
  %arrayidx134 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 %idxprom133
  %118 = load i64, ptr %arrayidx134, align 8
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.4, i64 noundef %118)
  %119 = load i32, ptr %eid129, align 4
  %cmp135 = icmp slt i32 %119, 25
  br i1 %cmp135, label %if.then136, label %if.end137

if.then136:                                       ; preds = %for.body132
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.5)
  br label %if.end137

if.end137:                                        ; preds = %if.then136, %for.body132
  br label %for.inc138

for.inc138:                                       ; preds = %if.end137
  %120 = load i32, ptr %eid129, align 4
  %inc139 = add nsw i32 %120, 1
  store i32 %inc139, ptr %eid129, align 4
  br label %for.cond130, !llvm.loop !12

for.end140:                                       ; preds = %for.cond130
  %arrayidx141 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 26
  %121 = load i64, ptr %arrayidx141, align 8
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.6, i64 noundef %121)
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.8)
  store i32 0, ptr %eid142, align 4
  br label %for.cond143

for.cond143:                                      ; preds = %for.inc154, %for.end140
  %122 = load i32, ptr %eid142, align 4
  %cmp144 = icmp slt i32 %122, 26
  br i1 %cmp144, label %for.body145, label %for.end156

for.body145:                                      ; preds = %for.cond143
  %123 = load i32, ptr %eid142, align 4
  %idxprom146 = sext i32 %123 to i64
  %arrayidx147 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 %idxprom146
  %124 = load i64, ptr %arrayidx147, align 8
  %125 = load i32, ptr %eid142, align 4
  %idxprom148 = sext i32 %125 to i64
  %arrayidx149 = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 %idxprom148
  %126 = load i64, ptr %arrayidx149, align 8
  %sub150 = sub i64 %124, %126
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.4, i64 noundef %sub150)
  %127 = load i32, ptr %eid142, align 4
  %cmp151 = icmp slt i32 %127, 25
  br i1 %cmp151, label %if.then152, label %if.end153

if.then152:                                       ; preds = %for.body145
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.5)
  br label %if.end153

if.end153:                                        ; preds = %if.then152, %for.body145
  br label %for.inc154

for.inc154:                                       ; preds = %if.end153
  %128 = load i32, ptr %eid142, align 4
  %inc155 = add nsw i32 %128, 1
  store i32 %inc155, ptr %eid142, align 4
  br label %for.cond143, !llvm.loop !13

for.end156:                                       ; preds = %for.cond143
  %arrayidx157 = getelementptr inbounds [27 x i64], ptr %after_hot_data, i64 0, i64 26
  %129 = load i64, ptr %arrayidx157, align 8
  %arrayidx158 = getelementptr inbounds [27 x i64], ptr %before_hot_data, i64 0, i64 26
  %130 = load i64, ptr %arrayidx158, align 8
  %sub159 = sub i64 %129, %130
  call void (ptr, ...) @hthread_printf(ptr noundef @.str.6, i64 noundef %sub159)
  br label %if.end160

if.end160:                                        ; preds = %for.end156, %for.end102, %if.then14
  ret void
}

declare i32 @get_group_size(...) #1

declare i32 @get_thread_id(...) #1

declare void @hthread_printf(ptr noundef, ...) #1

declare void @prof_start(i32 noundef) #1

declare i64 @prof_read(i32 noundef) #1

declare i64 @get_clk(...) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #2

declare i64 @prof_end(i32 noundef) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 20.1.6"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
