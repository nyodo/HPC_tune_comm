<template>
  <div class="comm-optimization-content">
    <el-card>
      <h4> 通信插桩调优工具 </h4>

      <el-form
        ref="commFormRef"
        :model="commForm"
        :rules="commRules"
        label-width="200px"
        size="small"
        status-icon
      >
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="作业名称" prop="name">
              <el-input v-model="commForm.name" placeholder="请输入作业名称" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="原始CSV文件（多选）" prop="raw_files">
              <el-upload
                ref="rawFileUpload"
                :auto-upload="false"
                :on-change="handleRawFileChange"
                :on-remove="handleRawFileRemove"
                :limit="500"
                accept=".csv"
                multiple
                drag
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
              </el-upload>
              <div v-if="commForm.raw_files.length" class="file-info">已上传 {{ commForm.raw_files.length }} 个文件</div>
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="通信类型" prop="comm_types">
              <el-input
                v-model="commForm.comm_typesText"
                placeholder="请输入通信类型编号，用逗号分隔，例如：55,56,57"
                clearable
              />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="测试轮数" prop="test_limit">
              <el-input-number v-model="commForm.test_limit" :min="1" :max="1000" :precision="0" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-form-item>
          <el-button type="primary" @click="startTuning" :loading="isRunning">
            {{ isRunning ? "调优中..." : "开始调优" }}
          </el-button>
          <el-button @click="resetCommForm">重置</el-button>
          <el-button @click="checkStatus" :disabled="!currentJobName">查看状态</el-button>
          <el-button @click="loadResults" :disabled="!currentJobName">查看结果</el-button>
        </el-form-item>
      </el-form>

      <el-card v-if="isRunning || outputLog.length > 0" style="margin-top: 20px">
        <template #header>
          <div style="display: flex; justify-content: space-between; align-items: center">
            <span>调优过程输出</span>
            <el-button size="small" @click="clearOutput">清空</el-button>
          </div>
        </template>
        <div class="output-container">
          <pre class="output-log">{{ outputLog }}</pre>
        </div>
      </el-card>

      <el-card v-if="resultData" style="margin-top: 20px">
        <template #header><span>调优结果</span></template>
        <div class="result-container">
          <el-descriptions :column="1" border>
            <el-descriptions-item label="最佳参数文件">
              <pre v-if="bestParamsPretty">{{ bestParamsPretty }}</pre>
              <span v-else>未找到最佳参数文件</span>
            </el-descriptions-item>
            <el-descriptions-item v-if="bestParamsMetrics" label="性能指标">
              <div class="metrics-grid">
                <div class="metric-item">
                  <div class="metric-label">默认耗时 (us)</div>
                  <div class="metric-value">{{ bestParamsMetrics.default_time_us }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">优化后耗时 (us)</div>
                  <div class="metric-value">{{ bestParamsMetrics.optimized_time_us }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">提升百分比 (%)</div>
                  <div class="metric-value">
                    {{ bestParamsMetrics.improvement_percent?.toFixed?.(2) ?? bestParamsMetrics.improvement_percent }}
                  </div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">加速比</div>
                  <div class="metric-value">{{ bestParamsMetrics.speedup }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">最佳轮次</div>
                  <div class="metric-value">{{ bestParamsMetrics.best_round }}</div>
                </div>
                <div class="metric-item">
                  <div class="metric-label">测试轮次</div>
                  <div class="metric-value">
                    {{ bestParamsMetrics.total_tests }} / {{ bestParamsMetrics.test_limit }}
                  </div>
                </div>
              </div>
            </el-descriptions-item>
            <el-descriptions-item label="结果文件列表">
              <pre>{{ resultData.file_list }}</pre>
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </el-card>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onUnmounted, computed } from "vue";
import { ElMessage } from "element-plus";
import { UploadFilled } from "@element-plus/icons-vue";
import axios from "axios";

const commFormRef = ref(null);
const rawFileUpload = ref(null);

const isRunning = ref(false);
const currentJobName = ref("");
const currentLogFilePath = ref("");
const currentResultDir = ref("");
const outputLog = ref("");
const resultData = ref(null);
let statusPollingTimer = null;

const bestParamsPretty = computed(() => {
  const raw = resultData.value?.best_params;
  if (!raw) return "";
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
});

const bestParamsMetrics = computed(() => {
  try {
    const parsed = resultData.value?.best_params ? JSON.parse(resultData.value.best_params) : null;
    return parsed?.metrics || null;
  } catch {
    return null;
  }
});

const commForm = reactive({
  name: "",
  raw_files: [],
  comm_typesText: "55",
  test_limit: 50,
});

const commRules = {
  name: [{ required: true, message: "请输入作业名称", trigger: "blur" }],
  raw_files: [{ required: true, type: "array", min: 1, message: "请至少上传 1 个 CSV 文件", trigger: "change" }],
  test_limit: [
    { required: true, type: "number", message: "请输入测试轮数", trigger: "change" },
    { type: "number", min: 1, max: 1000, message: "测试轮数范围：1~1000", trigger: "change" },
  ],
};

const handleRawFileChange = async (file) => {
  try {
    const formData = new FormData();
    formData.append("file", file.raw);
    const response = await axios.post("/api/upload-file", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    if (response.data.filePath) {
      commForm.raw_files.push({
        file_path: response.data.filePath,
        filename: file.name,
        file_id: response.data.file_id,
        uid: file.uid,
      });
      ElMessage.success(`上传成功：${file.name}`);
    }
  } catch (error) {
    ElMessage.error("文件上传失败：" + (error.response?.data?.error || error.message));
  }
};

const handleRawFileRemove = async (file) => {
  const idx = commForm.raw_files.findIndex((f) => f.uid === file.uid);
  if (idx >= 0) {
    const removed = commForm.raw_files[idx];
    commForm.raw_files.splice(idx, 1);
    // 尝试通知后端删除本地临时文件
    try {
      await axios.post(
        "/api/delete-file",
        { filePath: removed.file_path, file_id: removed.file_id },
        { headers: { "Content-Type": "application/json" } }
      );
    } catch (e) {
      // 忽略删除失败
    }
  }
};

const normalizeCommPayload = () => {
  const comm_types = commForm.comm_typesText
    .split(",")
    .map((c) => parseInt(c.trim()))
    .filter((c) => !isNaN(c));

  return {
    name: commForm.name,
    raw_files: commForm.raw_files.map(({ file_path, filename, file_id }) => ({ file_path, filename, file_id })),
    comm_types: comm_types.length > 0 ? comm_types : [55],
    test_limit: commForm.test_limit,

    // 兼容后端可能仍要求的字段：这里给出默认值，避免因前端删字段导致任务提交失败
    partition: "thcp3",
    nodes: [],
    raw_dir_name: "",
    processed_subdir: "",
    script_path: "",
    result_dir: "",
    save_shell_output: false,
    no_dups: true,
    opentuner_log_dir: "",
    save_opentuner_log: true,
  };
};

const startTuning = async () => {
  try {
    const ok = await commFormRef.value?.validate?.();
    if (!ok) return;

    if (!commForm.raw_files.length) {
      ElMessage.error("请先上传原始 CSV 文件");
      return;
    }

    const payload = normalizeCommPayload();
    isRunning.value = true;
    currentJobName.value = payload.name;
    outputLog.value = "正在提交调优任务...\n";

    const response = await axios.post("/api/comm-collection", payload, {
      headers: { "Content-Type": "application/json" },
    });

    if (response.data.status === "success") {
      ElMessage.success("调优任务已提交，开始执行...");
      outputLog.value += response.data.data.output || "";
      outputLog.value += "\n任务已启动，正在监控输出...\n";

      currentLogFilePath.value = response.data.data.log_file_path || "";
      currentResultDir.value = response.data.data.result_dir || "";

      startStatusPolling(currentJobName.value, currentLogFilePath.value);
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

const startStatusPolling = (jobName, logFilePath) => {
  if (statusPollingTimer) clearInterval(statusPollingTimer);
  statusPollingTimer = setInterval(async () => {
    try {
      const response = await axios.post(
        "/api/comm-status",
        { job_name: jobName, log_file_path: logFilePath },
        { headers: { "Content-Type": "application/json" } }
      );

      if (response.data.status === "success") {
        if (response.data.log_output) outputLog.value = response.data.log_output;
        if (!response.data.is_running) {
          clearInterval(statusPollingTimer);
          statusPollingTimer = null;
          isRunning.value = false;
          ElMessage.success("调优任务已完成");
          loadResults();
        }
      }
    } catch (error) {
      console.error("状态查询失败:", error);
    }
  }, 3000);
};

const checkStatus = async () => {
  if (!currentJobName.value) {
    ElMessage.warning("请先启动调优任务");
    return;
  }

  try {
    const response = await axios.post(
      "/api/comm-status",
      { job_name: currentJobName.value, log_file_path: currentLogFilePath.value },
      { headers: { "Content-Type": "application/json" } }
    );
    if (response.data.status === "success") {
      if (response.data.log_output) outputLog.value = response.data.log_output;
      ElMessage.success("状态已更新");
    }
  } catch (error) {
    ElMessage.error("状态查询失败：" + (error.response?.data?.message || error.message));
  }
};

const loadResults = async () => {
  if (!currentJobName.value) {
    ElMessage.warning("请先启动调优任务");
    return;
  }

  try {
    const response = await axios.post(
      "/api/comm-results",
      { result_dir: currentResultDir.value || commForm.result_dir },
      { headers: { "Content-Type": "application/json" } }
    );
    if (response.data.status === "success") {
      resultData.value = response.data;
      ElMessage.success("结果已加载");
    }
  } catch (error) {
    ElMessage.error("结果加载失败：" + (error.response?.data?.message || error.message));
  }
};

const clearOutput = () => {
  outputLog.value = "";
};

const resetCommForm = () => {
  commForm.name = "";
  commForm.raw_files = [];
  commForm.comm_typesText = "55";
  commForm.test_limit = 50;

  outputLog.value = "";
  resultData.value = null;
  currentJobName.value = "";
  currentLogFilePath.value = "";
  currentResultDir.value = "";

  rawFileUpload.value?.clearFiles?.();
  commFormRef.value?.clearValidate?.();
  ElMessage.info("表单已重置");
};

onUnmounted(() => {
  if (statusPollingTimer) {
    clearInterval(statusPollingTimer);
    statusPollingTimer = null;
  }
});
</script>

<style scoped>
.comm-optimization-content {
  padding: 20px;
}
.file-info {
  margin-top: 10px;
  padding: 8px;
  background-color: #f5f7fa;
  border-radius: 4px;
  font-size: 12px;
  color: #606266;
}
.output-container {
  max-height: 500px;
  overflow-y: auto;
  background-color: #1e1e1e;
  padding: 15px;
  border-radius: 4px;
}
.output-log {
  margin: 0;
  color: #d4d4d4;
  font-family: "Courier New", monospace;
  font-size: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
}
.result-container {
  padding: 10px;
}
.result-container pre {
  margin: 0;
  padding: 10px;
  background-color: #f5f7fa;
  border-radius: 4px;
  font-size: 12px;
  max-height: 300px;
  overflow-y: auto;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.metric-item {
  padding: 10px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  background: #fafafa;
}
.metric-label {
  font-size: 12px;
  color: #909399;
  margin-bottom: 4px;
}
.metric-value {
  font-weight: 600;
  color: #303133;
}
</style>
