<template>
  <div class="collection-content">
    <el-card>
      <h4>访存数据采集</h4>
      <p>这里显示访存数据采集的相关功能。</p>

      <el-form :model="form" label-width="150px">
        <el-form-item label="采集类型">
          <el-select v-model="form.type" placeholder="请选择采集类型">
            <el-option label="内存访问数据" value="access" />
            <el-option label="缓存命中率" value="cache" />
          </el-select>
        </el-form-item>

        <el-form-item label="采集时间范围">
          <el-date-picker
            v-model="form.dateRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="emit('start', 'memory')">开始采集</el-button>
          <el-button @click="emit('stop', 'memory')">停止采集</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
defineProps({
  form: {
    type: Object,
    required: true,
  },
});

const emit = defineEmits(["start", "stop"]);
</script>

<style scoped>
.collection-content {
  padding: 20px;
}

.collection-content h4 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #003f88;
}
</style>
