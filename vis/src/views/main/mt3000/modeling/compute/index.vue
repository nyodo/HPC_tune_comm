<template>
  <div class="modeling-content">
    <p>这里显示数据建模的相关功能。</p>

    <el-form :model="modelingForm" label-width="150px">
      <el-form-item label="模型类型">
        <el-select v-model="modelingForm.modelType" placeholder="请选择模型类型">
          <el-option label="性能预测模型" value="performance" />
          <el-option label="资源使用模型" value="resource" />
          <el-option label="负载均衡模型" value="loadbalance" />
        </el-select>
      </el-form-item>

      <el-form-item label="训练数据">
        <el-upload
          class="upload-demo"
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
          :file-list="modelingForm.fileList"
        >
          <el-button type="primary">选择文件</el-button>
          <template #tip>
            <div class="el-upload__tip">支持上传csv、txt等格式文件</div>
          </template>
        </el-upload>
      </el-form-item>

      <el-form-item>
        <el-button type="primary" @click="startModeling">开始建模</el-button>
        <el-button @click="resetModeling">重置</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup>
import { reactive } from "vue";
import { ElMessage } from "element-plus";

const modelingForm = reactive({
  modelType: "",
  fileList: [],
});

const handleFileChange = (file, fileList) => {
  modelingForm.fileList = fileList;
};

const startModeling = () => {
  ElMessage.success("开始数据建模");
};

const resetModeling = () => {
  modelingForm.modelType = "";
  modelingForm.fileList = [];
  ElMessage.info("已重置建模表单");
};
</script>

<style scoped>
.modeling-content {
  padding: 20px;
}

.modeling-content h4 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #003f88;
}

.upload-demo {
  width: 100%;
}
</style>
