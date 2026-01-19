<template>
  <div class="tab-content">
    <h3>目前超算状态</h3>
    <el-card class="status-card">
      <div class="status-content">
        <p>这里是超算状态监控页面，显示MT3000平台的实时运行状态。</p>
        <el-row :gutter="20">
          <el-col :span="6">
            <el-card shadow="hover">
              <div class="stat-item">
                <div class="stat-label">CPU使用率</div>
                <div class="stat-value">75%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <div class="stat-item">
                <div class="stat-label">内存使用率</div>
                <div class="stat-value">68%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <div class="stat-item">
                <div class="stat-label">网络带宽</div>
                <div class="stat-value">45%</div>
              </div>
            </el-card>
          </el-col>
          <el-col :span="6">
            <el-card shadow="hover">
              <div class="stat-item">
                <div class="stat-label">存储使用率</div>
                <div class="stat-value">52%</div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </div>
    </el-card>
  </div>
</template>

<script setup>
// 纯展示页：后续可在此接入真实接口
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

.status-card {
  margin-top: 20px;
}

.status-content {
  padding: 10px;
}

.stat-item {
  text-align: center;
  padding: 20px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 10px;
}

.stat-value {
  font-size: 32px;
  font-weight: bold;
  color: #003f88;
}
</style>
