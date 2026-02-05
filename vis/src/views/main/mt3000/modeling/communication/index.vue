<template>
  <div class="comm-modeling-content">
    <el-card>
      <h4>通信性能建模</h4>

      <el-form
        ref="modelingFormRef"
        :model="modelingForm"
        :rules="modelingRules"
        label-width="180px"
        size="small"
        status-icon
      >
        <el-divider content-position="left">数据选择</el-divider>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="建模任务名称" prop="name">
              <el-input v-model="modelingForm.name" placeholder="请输入建模任务名称" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="数据文件路径" prop="data_file">
              <el-input
                v-model="modelingForm.data_file"
                placeholder="例如：/path/to/comm_data.csv"
                clearable
              >
                <template #append>
                  <el-button @click="browseDataFiles">浏览</el-button>
                </template>
              </el-input>
              <div class="form-hint">请选择采集的通信性能数据文件（CSV格式）</div>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">模型配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="模型类型" prop="model_type">
              <el-select v-model="modelingForm.model_type" placeholder="请选择模型类型">
                <el-option label="线性回归" value="linear" />
                <el-option label="多项式回归" value="polynomial" />
                <el-option label="随机森林" value="random_forest" />
                <el-option label="神经网络" value="neural_network" />
              </el-select>
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="预测目标">
              <el-select v-model="modelingForm.target">
                <el-option label="延迟 (Latency)" value="latency" />
                <el-option label="带宽 (Bandwidth)" value="bandwidth" />
                <el-option label="吞吐量 (Throughput)" value="throughput" />
              </el-select>
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="特征列">
              <el-select v-model="modelingForm.features" multiple placeholder="请选择特征列">
                <el-option label="消息大小" value="message_size" />
                <el-option label="进程数" value="num_processes" />
                <el-option label="节点数" value="num_nodes" />
                <el-option label="通信模式" value="comm_pattern" />
                <el-option label="通信类型" value="comm_type" />
              </el-select>
              <div class="form-hint">选择用于建模的输入特征</div>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">数据划分</el-divider>
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="训练集比例">
              <el-slider v-model="modelingForm.train_ratio" :min="0.5" :max="0.9" :step="0.05" />
              <span>{{ (modelingForm.train_ratio * 100).toFixed(0) }}%</span>
            </el-form-item>
          </el-col>

          <el-col :span="8">
            <el-form-item label="验证集比例">
              <el-slider v-model="modelingForm.validation_ratio" :min="0.05" :max="0.3" :step="0.05" />
              <span>{{ (modelingForm.validation_ratio * 100).toFixed(0) }}%</span>
            </el-form-item>
          </el-col>

          <el-col :span="8">
            <el-form-item label="测试集比例">
              <el-slider v-model="modelingForm.test_ratio" :min="0.05" :max="0.3" :step="0.05" />
              <span>{{ (modelingForm.test_ratio * 100).toFixed(0) }}%</span>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">模型参数</el-divider>
        <el-row :gutter="20" v-if="modelingForm.model_type === 'polynomial'">
          <el-col :span="12">
            <el-form-item label="多项式阶数">
              <el-input-number v-model="modelingForm.model_params.degree" :min="1" :max="5" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20" v-if="modelingForm.model_type === 'random_forest'">
          <el-col :span="12">
            <el-form-item label="树的数量">
              <el-input-number v-model="modelingForm.model_params.n_estimators" :min="10" :max="500" :step="10" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="最大深度">
              <el-input-number v-model="modelingForm.model_params.max_depth" :min="1" :max="50" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-row :gutter="20" v-if="modelingForm.model_type === 'neural_network'">
          <el-col :span="12">
            <el-form-item label="隐藏层大小">
              <el-input v-model="modelingForm.model_params.hidden_layers" placeholder="例如：64,32,16" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="训练轮数">
              <el-input-number v-model="modelingForm.model_params.epochs" :min="10" :max="1000" :step="10" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">输出设置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="保存模型">
              <el-switch v-model="modelingForm.save_model" />
            </el-form-item>
          </el-col>

          <el-col :span="24" v-if="modelingForm.save_model">
            <el-form-item label="模型保存路径">
              <el-input v-model="modelingForm.model_output_path" placeholder="例如：models/comm_model.pkl" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider />
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item>
              <el-button type="primary" @click="submitModeling" :loading="isRunning">
                <el-icon><DataAnalysis /></el-icon>
                开始建模
              </el-button>
              <el-button @click="checkStatus" :disabled="!currentTaskId">
                <el-icon><Refresh /></el-icon>
                检查状态
              </el-button>
              <el-button @click="viewResults" :disabled="!modelMetrics || Object.keys(modelMetrics).length === 0">
                <el-icon><TrendCharts /></el-icon>
                查看结果
              </el-button>
              <el-button @click="resetForm">
                <el-icon><RefreshLeft /></el-icon>
                重置
              </el-button>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>

      <!-- 输出日志区域 -->
      <el-divider content-position="left">建模日志</el-divider>
      <el-card shadow="never" class="log-card">
        <pre class="log-output">{{ outputLog || "暂无日志输出..." }}</pre>
      </el-card>

      <!-- 模型性能指标 -->
      <el-divider content-position="left" v-if="modelMetrics && Object.keys(modelMetrics).length > 0">
        模型性能指标
      </el-divider>
      <el-row :gutter="20" v-if="modelMetrics && Object.keys(modelMetrics).length > 0">
        <el-col :span="8" v-for="(value, key) in modelMetrics" :key="key">
          <el-card shadow="hover">
            <el-statistic :title="key" :value="value" :precision="4" />
          </el-card>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, watch } from "vue";
import { ElMessage } from "element-plus";
import { DataAnalysis, Refresh, TrendCharts, RefreshLeft } from "@element-plus/icons-vue";
import axios from "axios";

const modelingFormRef = ref(null);
const isRunning = ref(false);
const outputLog = ref("");
const currentTaskId = ref(null);
const currentLogFilePath = ref("");
const modelMetrics = ref({});
let statusPollingTimer = null;

const modelingForm = reactive({
  name: "",
  data_file: "",
  model_type: "linear",
  features: ["message_size", "num_processes"],
  target: "latency",
  train_ratio: 0.8,
  validation_ratio: 0.1,
  test_ratio: 0.1,
  model_params: {
    degree: 2,
    n_estimators: 100,
    max_depth: 10,
    hidden_layers: "64,32",
    epochs: 100,
  },
  save_model: true,
  model_output_path: "models/comm_model.pkl",
});

const modelingRules = {
  name: [{ required: true, message: "请输入建模任务名称", trigger: "blur" }],
  data_file: [{ required: true, message: "请输入数据文件路径", trigger: "blur" }],
  model_type: [{ required: true, message: "请选择模型类型", trigger: "change" }],
};

// 监听数据划分比例，确保总和为1
watch(
  () => [modelingForm.train_ratio, modelingForm.validation_ratio, modelingForm.test_ratio],
  ([train, val, test]) => {
    const sum = train + val + test;
    if (Math.abs(sum - 1.0) > 0.01) {
      // 自动调整测试集比例
      modelingForm.test_ratio = Math.max(0.05, 1.0 - train - val);
    }
  }
);

const browseDataFiles = () => {
  ElMessage.info("浏览数据文件功能待实现");
  // TODO: 实现文件浏览功能
};

const submitModeling = async () => {
  try {
    await modelingFormRef.value.validate();
  } catch {
    ElMessage.warning("请填写完整表单");
    return;
  }

  try {
    isRunning.value = true;
    outputLog.value = "正在提交建模任务...\n";

    const response = await axios.post("/api/comm-modeling-submit", modelingForm, {
      headers: { "Content-Type": "application/json" },
    });

    if (response.data.status === "success") {
      ElMessage.success("建模任务已提交");
      outputLog.value += "任务已启动，正在训练模型...\n";

      currentTaskId.value = modelingForm.name;
      currentLogFilePath.value = response.data.data.log_file_path || "";

      startStatusPolling(currentLogFilePath.value);
    } else if (response.data.status === "warning") {
      ElMessage.warning(response.data.message);
      outputLog.value += `\n警告：${response.data.message}\n`;
      isRunning.value = false;
    } else {
      ElMessage.error(response.data.message || "任务提交失败");
      isRunning.value = false;
    }
  } catch (error) {
    const msg = error?.response?.data?.message || error?.message || "未知错误";
    ElMessage.error(`任务提交失败：${msg}`);
    isRunning.value = false;
    outputLog.value += `\n错误：${msg}\n`;
  }
};

const startStatusPolling = (logFilePath) => {
  if (statusPollingTimer) clearInterval(statusPollingTimer);
  statusPollingTimer = setInterval(async () => {
    try {
      const response = await axios.post(
        "/api/comm-modeling-status",
        { log_file_path: logFilePath },
        { headers: { "Content-Type": "application/json" } }
      );

      if (response.data.status === "success") {
        if (response.data.log_output) outputLog.value = response.data.log_output;
        if (response.data.model_metrics) modelMetrics.value = response.data.model_metrics;
        
        if (!response.data.is_running) {
          clearInterval(statusPollingTimer);
          statusPollingTimer = null;
          isRunning.value = false;
          ElMessage.success("建模任务已完成");
        }
      }
    } catch (error) {
      console.error("状态查询失败:", error);
    }
  }, 3000);
};

const checkStatus = async () => {
  if (!currentLogFilePath.value) {
    ElMessage.warning("没有正在运行的任务");
    return;
  }

  try {
    const response = await axios.post(
      "/api/comm-modeling-status",
      { log_file_path: currentLogFilePath.value },
      { headers: { "Content-Type": "application/json" } }
    );

    if (response.data.status === "success") {
      outputLog.value = response.data.log_output;
      if (response.data.model_metrics) modelMetrics.value = response.data.model_metrics;
      const status = response.data.is_running ? "运行中" : "已完成";
      ElMessage.info(`任务状态：${status}`);
    }
  } catch (error) {
    ElMessage.error("状态查询失败");
  }
};

const viewResults = () => {
  if (!modelMetrics.value || Object.keys(modelMetrics.value).length === 0) {
    ElMessage.warning("暂无模型性能指标");
    return;
  }
  ElMessage.success("模型性能指标已显示在下方");
};

const resetForm = () => {
  modelingFormRef.value.resetFields();
  outputLog.value = "";
  currentTaskId.value = null;
  currentLogFilePath.value = "";
  modelMetrics.value = {};
  if (statusPollingTimer) {
    clearInterval(statusPollingTimer);
    statusPollingTimer = null;
  }
  isRunning.value = false;
};
</script>

<style scoped>
.comm-modeling-content {
  padding: 20px;
}

.comm-modeling-content h4 {
  margin-top: 0;
  margin-bottom: 20px;
  color: #003f88;
}

.log-card {
  margin-top: 20px;
  background-color: #f5f7fa;
}

.log-output {
  max-height: 400px;
  overflow-y: auto;
  padding: 10px;
  font-size: 12px;
  line-height: 1.5;
  color: #333;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.form-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
</style>
