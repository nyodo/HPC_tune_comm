<template>
  <div class="tab-content">
    <h3>数据采集 (DCU)</h3>

    <el-card>
      <el-tabs v-model="active" type="card" class="nested-tabs">
        <el-tab-pane label="计算" name="compute">
          <CollectionCompute />
        </el-tab-pane>
        <el-tab-pane label="访存" name="memory">
          <CollectionMemory />
        </el-tab-pane>
        <el-tab-pane label="通信" name="communication">
          <CollectionCommunication />
        </el-tab-pane>
        <el-tab-pane label="I/O" name="io">
          <CollectionIO />
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from "vue";

import CollectionCompute from "./collection/compute/index.vue";
import CollectionMemory from "./collection/memory/index.vue";
import CollectionCommunication from "./collection/communication/index.vue";
import CollectionIO from "./collection/io/index.vue";

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
