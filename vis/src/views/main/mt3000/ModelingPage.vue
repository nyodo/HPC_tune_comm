<template>
  <div class="tab-content">
    <h3>数据建模</h3>

    <el-card>
      <el-tabs v-model="active" type="card" class="nested-tabs">
        <el-tab-pane label="计算" name="compute">
          <ModelingCompute />
        </el-tab-pane>
        <el-tab-pane label="访存" name="memory">
          <ModelingMemory />
        </el-tab-pane>
        <el-tab-pane label="通信" name="communication">
          <ModelingCommunication />
        </el-tab-pane>
        <el-tab-pane label="I/O" name="io">
          <ModelingIO />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from "vue";

import ModelingCompute from "./modeling/compute/index.vue";
import ModelingMemory from "./modeling/memory/index.vue";
import ModelingCommunication from "./modeling/communication/index.vue";
import ModelingIO from "./modeling/io/index.vue";

const active = ref("compute");
</script>

<style scoped>
.tab-content {
  padding: 20px;
}

.tab-content h3 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #003f88;
  font-size: 24px;
}

.nested-tabs {
  margin-top: 10px;
}
</style>
