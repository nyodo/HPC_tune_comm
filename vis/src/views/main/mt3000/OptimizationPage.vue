<template>
  <div class="tab-content">
    <h3>数据调优</h3>

    <el-card>
      <el-tabs v-model="active" type="card" class="nested-tabs">
        <el-tab-pane label="计算" name="compute">
          <OptCompute />
        </el-tab-pane>
        <el-tab-pane label="访存" name="memory">
          <OptMemory />
        </el-tab-pane>
        <el-tab-pane label="通信" name="communication">
          <OptCommunication />
        </el-tab-pane>
        <el-tab-pane label="I/O" name="io">
          <OptIO />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from "vue";

import OptCompute from "./optimization/compute/index.vue";
import OptMemory from "./optimization/memory/index.vue";
import OptCommunication from "./optimization/communication/index.vue";
import OptIO from "./optimization/io/index.vue";

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
