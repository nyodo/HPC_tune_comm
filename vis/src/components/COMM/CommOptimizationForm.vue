<template>
  <div class="comm-optimization-content">
    <el-card>
      <h4>通信调优（UCX with OpenTuner）</h4>

      <el-form
        ref="commFormRef"
        :model="commForm"
        :rules="commRules"
        label-width="200px"
        size="small"
        status-icon
      >
        <el-divider content-position="left">基本配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="作业名称" prop="name">
              <el-input v-model="commForm.name" placeholder="请输入作业名称" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="分区" prop="partition">
              <el-input v-model="commForm.partition" placeholder="例如：thcp3" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="节点列表" prop="nodesText">
              <el-input
                v-model="commForm.nodesText"
                placeholder="请输入节点名称，用逗号分隔，例如：cn27584,cn27585"
                clearable
              />
              <div class="form-hint">至少需要2个节点（用于跨节点通信测试）</div>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">数据文件</el-divider>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="原始数据目录名" prop="raw_dir_name">
              <el-input
                v-model="commForm.raw_dir_name"
                placeholder='例如：astro_demo_16_node（将保存为 data/app_data/astro_demo_16_node）'
                clearable
              />
              <div class="form-hint">
                生成的 config 会写入：raw_file_dir = <b>data/app_data/{{ commForm.raw_dir_name || "..." }}</b>
              </div>
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="processed 子目录" prop="processed_subdir">
              <el-input
                v-model="commForm.processed_subdir"
                placeholder="例如：lammps（将写入 data/processed/lammps/目录名.csv）"
                clearable
              />
              <div class="form-hint">
                生成的 config 会写入：csv_file =
                <b>data/processed/{{ commForm.processed_subdir || "..." }}/{{ commForm.raw_dir_name || "..." }}.csv</b>
                （由调优程序生成临时数据，不需要用户上传）
              </div>
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
                <template #tip>
                  <div class="el-upload__tip">
                    支持一次上传多份 CSV。后端会先保存到本机，再批量上传到远端
                    <b>data/app_data/{{ commForm.raw_dir_name || "目录名" }}/</b>
                  </div>
                </template>
              </el-upload>
              <div v-if="commForm.raw_files.length" class="file-info">
                已上传 {{ commForm.raw_files.length }} 个文件
              </div>
            </el-form-item>
          </el-col>

          <el-col :span="24">
            <el-form-item label="通信类型" prop="comm_types">
              <el-input
                v-model="commForm.comm_typesText"
                placeholder="请输入通信类型编号，用逗号分隔，例如：55,56,57"
                clearable
              />
              <div class="form-hint">只处理CSV文件中comm_type列匹配这些值的记录</div>
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">脚本配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="脚本路径">
              <el-input
                v-model="commForm.script_path"
                placeholder="例如：src/scripts/run_latency_2_intra-Blade.sh"
                clearable
              />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="结果目录">
              <el-input v-model="commForm.result_dir" placeholder="例如：result/result_2_Intra-Blade" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="保存Shell输出">
              <el-switch v-model="commForm.save_shell_output" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>
        </el-row>

        <el-divider content-position="left">OpenTuner调优配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="测试轮数" prop="test_limit">
              <el-input-number v-model="commForm.test_limit" :min="1" :max="1000" :precision="0" />
              <div class="form-hint">调优程序将执行的测试轮数</div>
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="去重">
              <el-switch v-model="commForm.no_dups" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="日志目录">
              <el-input v-model="commForm.opentuner_log_dir" placeholder="例如：tune_result_avg" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="保存OpenTuner日志">
              <el-switch v-model="commForm.save_opentuner_log" active-text="是" inactive-text="否" />
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
  partition: "thcp3",
  nodesText: "cn27584,cn27585",
  raw_dir_name: "astro_demo_16_node",
  processed_subdir: "lammps",
  raw_files: [],
  comm_typesText: "55",
  script_path: "src/scripts/run_latency_2_intra-Blade.sh",
  result_dir: "result/result_2_Intra-Blade",
  save_shell_output: false,
  test_limit: 50,
  no_dups: true,
  opentuner_log_dir: "tune_result_avg",
  save_opentuner_log: true,
});

const commRules = {
  name: [{ required: true, message: "请输入作业名称", trigger: "blur" }],
  partition: [{ required: true, message: "请输入分区名称", trigger: "blur" }],
  nodesText: [{ required: true, message: "请输入节点列表", trigger: "blur" }],
  raw_dir_name: [{ required: true, message: "请输入原始数据目录名", trigger: "blur" }],
  processed_subdir: [{ required: true, message: "请输入 processed 子目录", trigger: "blur" }],
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
  const nodes = commForm.nodesText
    .split(",")
    .map((n) => n.trim())
    .filter(Boolean);

  const comm_types = commForm.comm_typesText
    .split(",")
    .map((c) => parseInt(c.trim()))
    .filter((c) => !isNaN(c));

  return {
    name: commForm.name,
    partition: commForm.partition,
    nodes,
    raw_dir_name: commForm.raw_dir_name,
    processed_subdir: commForm.processed_subdir,
    raw_files: commForm.raw_files.map(({ file_path, filename, file_id }) => ({ file_path, filename, file_id })),
    comm_types: comm_types.length > 0 ? comm_types : [55],
    script_path: commForm.script_path,
    result_dir: commForm.result_dir,
    save_shell_output: commForm.save_shell_output,
    test_limit: commForm.test_limit,
    no_dups: commForm.no_dups,
    opentuner_log_dir: commForm.opentuner_log_dir,
    save_opentuner_log: commForm.save_opentuner_log,
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
  commForm.partition = "thcp3";
  commForm.nodesText = "cn27584,cn27585";
  commForm.raw_dir_name = "astro_demo_16_node";
  commForm.processed_subdir = "lammps";
  commForm.raw_files = [];
  commForm.comm_typesText = "55";
  commForm.script_path = "src/scripts/run_latency_2_intra-Blade.sh";
  commForm.result_dir = "result/result_2_Intra-Blade";
  commForm.save_shell_output = false;
  commForm.test_limit = 50;
  commForm.no_dups = true;
  commForm.opentuner_log_dir = "tune_result_avg";
  commForm.save_opentuner_log = true;

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
.form-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
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
