<template>
  <div class="tab-content">
    <h3>数据采集</h3>

    <el-tabs v-model="active" type="card" class="nested-tabs">
      <el-tab-pane label="计算数据采集" name="compute">
        <CollectionCompute :form="computeForm" @start="start" @stop="stop" />
      </el-tab-pane>

      <el-tab-pane label="访存数据采集" name="memory">
        <CollectionMemory :form="memoryForm" @start="start" @stop="stop" />
      </el-tab-pane>

      <el-tab-pane label="通信数据采集" name="communication">
        <CollectionCommunication :form="communicationForm" @start="start" @stop="stop" />
      </el-tab-pane>

      <el-tab-pane label="I/O数据采集" name="io">
        <CollectionIO />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref, reactive } from "vue";
import { ElMessage } from "element-plus";

import CollectionCompute from "./collection/compute/index.vue";
import CollectionMemory from "./collection/memory/index.vue";
import CollectionCommunication from "./collection/communication/index.vue";
import CollectionIO from "./collection/io/index.vue";

const active = ref("compute");

const computeForm = reactive({
  type: "",
  dateRange: [],
});

const memoryForm = reactive({
  type: "",
  dateRange: [],
});

const communicationForm = reactive({
  type: "",
  dateRange: [],
});

const start = (type) => {
  ElMessage.success(`开始${type}数据采集`);
};

const stop = (type) => {
  ElMessage.info(`停止${type}数据采集`);
};
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
  margin-top: 20px;
}
</style>
