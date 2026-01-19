<template>
  <div class="dcu-container">
    <div class="page-header">
      <h2>DCU超算平台</h2>
      <el-button class="back-btn" type="primary" plain size="small" @click="goBack">返回平台选择</el-button>
    </div>

    <el-tabs v-model="activeTab" type="border-card" class="platform-tabs">
      <el-tab-pane label="目前超算状态" name="status">
        <StatusPage />
      </el-tab-pane>

      <el-tab-pane label="数据采集" name="collection">
        <CollectionPage />
      </el-tab-pane>

      <el-tab-pane label="数据建模" name="modeling">
        <ModelingPage />
      </el-tab-pane>

      <el-tab-pane label="数据调优" name="optimization">
        <OptimizationPage />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";

import StatusPage from "./dcu/StatusPage.vue";
import CollectionPage from "./dcu/CollectionPage.vue";
import ModelingPage from "./dcu/ModelingPage.vue";
import OptimizationPage from "./dcu/OptimizationPage.vue";

const router = useRouter();
const activeTab = ref("status");

const goBack = () => {
  router.push("/main");
};
</script>

<style scoped>
.dcu-container {
  padding: 20px;
  background-color: #f0f2f5;
  min-height: calc(100vh - 70px);
}

.page-header {
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.back-btn {
  margin-left: 12px;
}

.page-header h2 {
  margin: 0;
  color: #003f88;
  font-size: 28px;
  font-weight: bold;
}

.platform-tabs {
  background-color: white;
  border-radius: 8px;
}
</style>
