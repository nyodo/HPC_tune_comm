<template>
  <div class="comm-optimization-wrapper">
    <el-tabs type="border-card">
      <el-tab-pane label="通信插桩调优工具">
        <CommOptimizationForm />
      </el-tab-pane>
      <el-tab-pane label="主要调优工具">
        <div class="placeholder-content">
          <el-empty description="主要调优工具即将上线，敬请期待" />
        </div>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import CommOptimizationForm from "@/components/COMM/CommOptimizationForm.vue";
</script>

<style scoped>
.comm-optimization-wrapper {
  padding: 20px;
}
.placeholder-content {
  padding: 40px 0;
}
</style>
