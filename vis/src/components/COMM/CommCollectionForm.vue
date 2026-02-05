<template>
  <div class="comm-collection-content">
    <el-card>
      <h4>通信数据采集</h4>

      <el-form
        ref="commFormRef"
        :model="commForm"
        :rules="commRules"
        label-width="200px"
        size="small"
        status-icon
      >
        <!-- 基本配置 -->
        <el-divider content-position="left">基本配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="作业名称" prop="name">
              <el-input v-model="commForm.name" placeholder="请输入作业名称" clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="分区" prop="partition">
              <el-input v-model="commForm.partition" placeholder="例如：normal" clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="节点数" prop="nodes">
              <el-input-number v-model="commForm.nodes" :min="1" :max="256" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="每节点进程数" prop="ntasksPerNode">
              <el-input-number v-model="commForm.ntasksPerNode" :min="1" :max="64" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="内存" prop="mem">
              <el-input v-model="commForm.mem" placeholder="例如：100G" clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="GPU/DCU" prop="gres">
              <el-input v-model="commForm.gres" placeholder="例如：dcu:4" clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="时限（可选）">
              <el-input v-model="commForm.timeLimit" placeholder="例如：02:00:00" clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="独占节点">
              <el-switch v-model="commForm.exclusive" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="禁止重排">
              <el-switch v-model="commForm.noRequeue" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 目录与程序 -->
        <el-divider content-position="left">目录与程序</el-divider>
        <el-row :gutter="20">
          <el-col :span="24">
            <el-form-item label="应用目录" prop="appHome">
              <el-input v-model="commForm.appHome" placeholder="例如：/work1/dtune/zhengt/..." clearable />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="入口程序" prop="appEntry">
              <el-input v-model="commForm.appEntry" placeholder="默认：./app-run.sh" clearable />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 插桩与采集 -->
        <el-divider content-position="left">插桩与采集</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="失败时清理日志">
              <el-switch v-model="commForm.cleanupOnFail" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 环境模块 -->
        <el-divider content-position="left">环境模块</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="模块策略">
              <el-select v-model="commForm.moduleStrategy">
                <el-option label="预设" value="preset" />
                <el-option label="自定义" value="custom" />
              </el-select>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 预设模块（只读） -->
        <el-row v-if="commForm.moduleStrategy === 'preset'">
          <el-col :span="24">
            <el-form-item label="预设模块">
              <el-card shadow="hover" class="params-card">
                <div v-for="(mod, idx) in commForm.presetModules" :key="idx" class="param-row">
                  <el-input :value="`module load ${mod}`" readonly />
                </div>
              </el-card>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 自定义模块（动态行） -->
        <el-row v-if="commForm.moduleStrategy === 'custom'">
          <el-col :span="24">
            <el-form-item label="自定义模块">
              <el-card shadow="hover" class="params-card">
                <div v-for="(mod, idx) in commForm.customModules" :key="idx" class="param-row">
                  <el-input v-model="mod.moduleText" placeholder="例如：compiler/rocm/dtk/25.04" clearable />
                  <el-button type="danger" plain size="small" @click="removeCustomModule(idx)">删除</el-button>
                </div>
                <div class="param-row-actions">
                  <el-button size="small" @click="addCustomModule">添加模块</el-button>
                  <el-button size="small" type="warning" plain @click="clearCustomModules">清空</el-button>
                </div>
              </el-card>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 模块移除（动态行） -->
        <el-row>
          <el-col :span="24">
            <el-form-item label="移除模块">
              <el-card shadow="hover" class="params-card">
                <div v-for="(_, idx) in commForm.moduleRemoves" :key="idx" class="param-row">
                  <el-input v-model="commForm.moduleRemoves[idx]" placeholder="例如：mathlib/fftw/3.3.8/double/gnu" clearable />
                  <el-button type="danger" plain size="small" @click="removeModuleRemove(idx)">删除</el-button>
                </div>
                <div class="param-row-actions">
                  <el-button size="small" @click="addModuleRemove">添加移除项</el-button>
                  <el-button size="small" type="warning" plain @click="clearModuleRemoves">清空</el-button>
                </div>
              </el-card>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 环境变量 -->
        <el-divider content-position="left">环境变量（export）</el-divider>
        <el-row>
          <el-col :span="24">
            <el-form-item label="环境变量">
              <el-card shadow="hover" class="params-card">
                <div v-for="(env, idx) in commForm.envExports" :key="idx" class="param-row">
                  <el-input v-model="env.key" placeholder="变量名，例如：UCX_TLS" clearable />
                  <el-input v-model="env.value" placeholder="变量值，例如：self,sm,rc_x" clearable />
                  <el-button type="danger" plain size="small" @click="removeEnvExport(idx)">删除</el-button>
                </div>
                <div class="param-row-actions">
                  <el-button size="small" @click="addEnvExport">添加环境变量</el-button>
                  <el-button size="small" type="warning" plain @click="clearEnvExports">清空</el-button>
                </div>
              </el-card>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- mpirun 参数 -->
        <el-divider content-position="left">mpirun 参数</el-divider>
        <el-row>
          <el-col :span="24">
            <el-form-item label="mpirun 参数">
              <el-card shadow="hover" class="params-card">
                <div v-for="(arg, idx) in commForm.mpirunArgs" :key="idx" class="param-row">
                  <el-input v-model="arg.key" placeholder="参数名，例如：-np 或 --mca" clearable />
                  <el-input v-model="arg.value" placeholder="参数值（可空），例如：${SLURM_NPROCS} 或 plm_rsh_no_tree_spawn 1" clearable />
                  <el-button type="danger" plain size="small" @click="removeMpirunArg(idx)">删除</el-button>
                </div>
                <div class="param-row-actions">
                  <el-button size="small" @click="addMpirunArg">添加参数</el-button>
                  <el-button size="small" type="warning" plain @click="clearMpirunArgs">清空</el-button>
                </div>
              </el-card>
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 操作区 -->
        <el-form-item>
          <el-button type="primary" @click="generateScript">生成脚本预览</el-button>
          <el-button type="primary" :loading="isRunning" @click="submitCollection">提交到 MT-3000</el-button>
          <el-button @click="checkStatus" :disabled="!currentLogFilePath">检查状态</el-button>
          <el-button @click="loadResults" :disabled="!currentResultDir">查看结果</el-button>
          <el-button @click="saveDraft">保存草稿</el-button>
          <el-button @click="loadDraft">加载草稿</el-button>
          <el-button @click="exportConfigJson">导出配置</el-button>
          <el-button @click="resetForm">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 脚本预览区 -->
    <el-card v-if="generatedScript" style="margin-top: 20px">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>生成的 sbatch 脚本</span>
          <div>
            <el-button size="small" @click="copyScript">复制</el-button>
            <el-button size="small" @click="downloadScript">下载</el-button>
          </div>
        </div>
      </template>
      <div class="script-container">
        <pre class="generated-script">{{ generatedScript }}</pre>
      </div>
    </el-card>

    <!-- 任务状态区 -->
    <el-card style="margin-top: 20px">
      <template #header><span>任务状态</span></template>
      <el-form label-width="120px" size="small">
        <el-form-item label="任务">
          <el-tag v-if="currentTaskId">{{ currentTaskId }}</el-tag>
          <el-tag v-else type="info">未提交</el-tag>
        </el-form-item>
        <el-form-item label="状态">
          <el-tag :type="isRunning ? 'warning' : (currentTaskId ? 'success' : 'info')">
            {{ isRunning ? '运行中' : (currentTaskId ? '已结束' : '未提交') }}
          </el-tag>
        </el-form-item>
        <el-form-item label="下载">
          <div style="display: flex; gap: 10px;">
            <el-button
              size="small"
              type="primary"
              :disabled="!currentLogFilePath"
              @click="downloadInstrumentationLog"
            >
              下载插桩日志
            </el-button>
            <el-button
              size="small"
              type="primary"
              :disabled="!currentLogFilePath"
              @click="downloadProgramOutput"
            >
              下载程序输出
            </el-button>
          </div>
        </el-form-item>
        <el-form-item label="日志">
          <el-input type="textarea" :rows="10" readonly :value="outputLog || '暂无日志输出...'" />
        </el-form-item>
      </el-form>
    </el-card>

    <!-- 结果文件 -->
    <el-card style="margin-top: 20px" v-if="resultFiles.length">
      <template #header><span>采集结果文件</span></template>
      <el-table :data="resultFiles" border>
        <el-table-column prop="filename" label="文件名" />
        <el-table-column prop="size" label="大小" width="140" />
        <el-table-column label="路径" prop="path" />
        <el-table-column label="操作" width="120">
          <template #default="scope">
            <el-button v-if="scope.row.filename === 'outline.csv'" size="small" type="primary" @click="parseOutline(scope.row.path)">解析 outline</el-button>
            <span v-else style="color: #999; font-size: 12px;">-</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 解析结果：outline -->
    <el-card style="margin-top: 20px" v-if="parsedOutline">
      <template #header><span>outline.csv 解析结果</span></template>
      <el-row :gutter="20">
        <el-col :span="6" v-for="(v, k) in parsedOutline.meta" :key="k">
          <el-statistic :title="k" :value="v" />
        </el-col>
      </el-row>
      <el-divider />
      <el-row :gutter="20">
        <el-col :span="8">
          <el-statistic title="总通信时间（最小）" :value="parsedOutline.stats.min" suffix="us" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="总通信时间（平均）" :value="parsedOutline.stats.avg" suffix="us" />
        </el-col>
        <el-col :span="8">
          <el-statistic title="总通信时间（最大）" :value="parsedOutline.stats.max" suffix="us" />
        </el-col>
      </el-row>
      <el-divider />
      <div id="outline-chart" style="width: 100%; height: 400px;"></div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, watch, onBeforeUnmount, nextTick } from "vue";
import { ElMessage } from "element-plus";
import { generateSubScript, normalizeCommCollectPayload } from "@/utils/commCollectScript.js";
import axios from "axios";
import * as echarts from "echarts";

const commFormRef = ref(null);
const generatedScript = ref("");

const isRunning = ref(false);
const outputLog = ref("");
const currentTaskId = ref(null);
const resultFiles = ref([]);
const parsedOutline = ref(null);
const parsedLogSummary = ref(null);
let statusPollingTimer = null;

async function downloadInstrumentationLog() {
  if (!currentTaskId.value) return ElMessage.warning("没有任务ID");
  await downloadFileFromBackend("/comm-collection-download-instrumentation-log", {
    job_name: currentTaskId.value,
  });
}

async function downloadProgramOutput() {
  if (!currentTaskId.value) return ElMessage.warning("没有任务ID");
  await downloadFileFromBackend("/comm-collection-download-program-output", {
    job_name: currentTaskId.value,
  });
}

async function downloadFileFromBackend(url, payload) {
  try {
    const resp = await axios.post(url, payload, {
      responseType: "blob",
      headers: { "Content-Type": "application/json" },
      validateStatus: () => true,
    });

    if (resp.status !== 200) {
      let msg = `下载失败(${resp.status})`;
      try {
        const text = await resp.data.text();
        msg += text ? `: ${text}` : "";
      } catch {}
      ElMessage.error(msg);
      return;
    }

    const cd = resp.headers?.["content-disposition"] || resp.headers?.get?.("content-disposition");
    let filename = "download.bin";
    if (typeof cd === "string") {
      const m = cd.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
      filename = decodeURIComponent(m?.[1] || m?.[2] || filename);
    }

    const blob = resp.data instanceof Blob ? resp.data : new Blob([resp.data]);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (e) {
    ElMessage.error("下载失败：" + (e?.message || "未知错误"));
  }
}

const initForm = () => ({
  name: "",
  partition: "normal",
  nodes: 16,
  ntasksPerNode: 9,
  mem: "100G",
  gres: "dcu:4",
  exclusive: true,
  noRequeue: true,
  timeLimit: "",
  appHome: "",
  appEntry: "./app-run.sh",
  cleanupOnFail: true,
  moduleStrategy: "preset",
  presetModules: [
    "compiler/rocm/dtk/25.04",
    "compiler/devtoolset/7.3.1",
    "mpi/hpcx/2.4.1/gcc-7.3.1",
    "compiler/cmake/3.24.1",
  ],
  customModules: [{ moduleText: "" }],
  moduleRemoves: [
    "mathlib/fftw/3.3.8/double/gnu",
    "mathlib/fftw/3.3.8/single/gnu",
  ],
  envExports: [
    { key: "UCX_TLS", value: "self,sm,rc_x" },
    { key: "UCX_RC_VERBS_TIMEOUT", value: "5000000.00us" },
    { key: "UCX_RC_VERBS_RNR_TIMEOUT", value: "60000.00us" },
  ],
  mpirunArgs: [
    { key: "-np", value: "${SLURM_NPROCS}" },
    { key: "--rankfile", value: "./rankfile" },
    { key: "--leave-session-attached", value: "" },
    { key: "--mca", value: "plm_rsh_no_tree_spawn 1" },
    { key: "--mca", value: "plm_rsh_num_concurrent ${SLURM_JOB_NUM_NODES}" },
    { key: "-mca", value: "routed_radix ${SLURM_JOB_NUM_NODES}" },
    { key: "-mca", value: "pml ucx" },
  ],
});

const commForm = reactive(initForm());

const commRules = {
  name: [{ required: true, message: "请输入作业名称", trigger: "blur" }],
  partition: [{ required: true, message: "请输入分区名", trigger: "blur" }],
  nodes: [{ required: true, type: "number", min: 1, max: 256, trigger: "blur" }],
  ntasksPerNode: [{ required: true, type: "number", min: 1, max: 64, trigger: "blur" }],
  mem: [{ required: true, message: "请输入内存配置", trigger: "blur" }],
  gres: [{ required: true, message: "请输入 GPU/DCU 配置", trigger: "blur" }],
  appHome: [{ required: true, message: "请输入应用目录", trigger: "blur" }],
  appEntry: [{ required: true, message: "请输入入口程序", trigger: "blur" }],
};

function addCustomModule() { commForm.customModules.push({ moduleText: "" }); }
function removeCustomModule(idx) { commForm.customModules.splice(idx, 1); if (!commForm.customModules.length) commForm.customModules = [{ moduleText: "" }]; }
function clearCustomModules() { commForm.customModules = [{ moduleText: "" }]; }
function addModuleRemove() { commForm.moduleRemoves.push(""); }
function removeModuleRemove(idx) { commForm.moduleRemoves.splice(idx, 1); }
function clearModuleRemoves() { commForm.moduleRemoves = []; }
function addEnvExport() { commForm.envExports.push({ key: "", value: "" }); }
function removeEnvExport(idx) { commForm.envExports.splice(idx, 1); if (!commForm.envExports.length) commForm.envExports = [{ key: "", value: "" }]; }
function clearEnvExports() { commForm.envExports = [{ key: "", value: "" }]; }
function addMpirunArg() { commForm.mpirunArgs.push({ key: "", value: "" }); }
function removeMpirunArg(idx) { commForm.mpirunArgs.splice(idx, 1); if (!commForm.mpirunArgs.length) commForm.mpirunArgs = [{ key: "", value: "" }]; }
function clearMpirunArgs() { commForm.mpirunArgs = [{ key: "", value: "" }]; }

function generateScript() {
  try {
    commFormRef.value?.validate?.();
    const cfg = normalizeCommCollectPayload(commForm);
    generatedScript.value = generateSubScript(cfg);
    ElMessage.success("脚本已生成");
  } catch (e) {
    ElMessage.error("脚本生成失败：" + e.message);
  }
}
function copyScript() { navigator.clipboard.writeText(generatedScript.value).then(() => ElMessage.success("已复制到剪贴板")); }
function downloadScript() {
  const blob = new Blob([generatedScript.value], { type: "text/x-shellscript" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${commForm.name || "comm_collect"}.sh`;
  a.click();
  URL.revokeObjectURL(url);
}

function saveDraft() { localStorage.setItem("comm_collect_draft", JSON.stringify(commForm)); ElMessage.success("草稿已保存"); }
function loadDraft() {
  const raw = localStorage.getItem("comm_collect_draft");
  if (!raw) return ElMessage.warning("没有找到草稿");
  Object.assign(commForm, JSON.parse(raw));
  ElMessage.success("草稿已加载");
}
function exportConfigJson() {
  const cfg = normalizeCommCollectPayload(commForm);
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${commForm.name || "comm_collect"}_config.json`;
  a.click();
  URL.revokeObjectURL(url);
  ElMessage.success("配置已导出");
}

function stopStatusPolling() { if (statusPollingTimer) { clearInterval(statusPollingTimer); statusPollingTimer = null; } }

async function submitCollection() {
  try {
    await commFormRef.value?.validate?.();
  } catch {
    ElMessage.warning("请填写完整表单");
    return;
  }

  try {
    isRunning.value = true;
    outputLog.value = "正在提交通信采集任务...\n";

    const payload = { ...normalizeCommCollectPayload(commForm), step: "1" };
    const response = await axios.post("/comm-collection-submit", payload, { headers: { "Content-Type": "application/json" } });

    if (response.data?.status === "success") {
      const data = response.data?.data || {};
      currentTaskId.value = data.job_name || payload.name || null;
      if (data.remote_cmd) outputLog.value += `远程命令：\n${data.remote_cmd}\n\n`;
      outputLog.value += "任务已启动，正在监控日志...\n";
      startStatusPolling(currentTaskId.value);
      ElMessage.success("采集任务已提交");
    } else {
      throw new Error(response.data?.message || "任务提交失败");
    }
  } catch (e) {
    ElMessage.error("任务提交失败：" + (e?.message || "未知错误"));
    outputLog.value += `\n错误：${e?.message || "未知错误"}\n`;
    isRunning.value = false;
  }
}

function startStatusPolling(jobName) {
  stopStatusPolling();
  statusPollingTimer = setInterval(async () => {
    try {
      const response = await axios.post(
        "/comm-collection-status",
        { job_name: jobName },
        { headers: { "Content-Type": "application/json" } }
      );
      if (response.data?.status === "success") {
        if (typeof response.data.log_output === "string") outputLog.value = response.data.log_output;
        if (!response.data.is_running) {
          stopStatusPolling();
          isRunning.value = false;
          ElMessage.success("采集任务已完成");
          await loadResults();
        }
      }
    } catch (err) {
      console.error("状态查询失败:", err);
    }
  }, 3000);
}

async function checkStatus() {
  if (!currentTaskId.value) return ElMessage.warning("没有正在运行的任务");
  const response = await axios.post(
    "/comm-collection-status",
    { job_name: currentTaskId.value },
    { headers: { "Content-Type": "application/json" } }
  );
  if (response.data?.status === "success") {
    outputLog.value = response.data.log_output;
    ElMessage.info(`任务状态：${response.data.is_running ? "运行中" : "已完成"}`);
  }
}

async function loadResults() {
  if (!currentTaskId.value) return ElMessage.warning("没有任务ID");
  const response = await axios.post(
    "/comm-collection-results",
    { job_name: currentTaskId.value },
    { headers: { "Content-Type": "application/json" } }
  );
  if (response.data?.status === "success") {
    resultFiles.value = response.data.result_files || [];
    const outline = resultFiles.value.find((f) => f.filename === "outline.csv");
    if (outline?.path) await parseOutline(outline.path, { silent: true });
    ElMessage.success(`找到 ${resultFiles.value.length} 个结果文件`);
  }
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] !== undefined ? sorted[base] + rest * (sorted[base + 1] - sorted[base]) : sorted[base];
}

async function parseOutline(filePath, opts = {}) {
  try {
    const resp = await axios.post("/comm-collection-read-file", { file_path: filePath }, { headers: { "Content-Type": "application/json" } });
    if (resp.data?.status !== "success") throw new Error(resp.data?.message || "读取 outline.csv 失败");

    const lines = String(resp.data.content || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const meta = {};
    const totals = [];

    for (const line of lines) {
      const parts = line.split(",");
      if (parts.length === 2 && parts[0] && !/^[-+]?\d/.test(parts[0])) {
        meta[parts[0]] = parts[1];
        continue;
      }
      const nums = parts.map((p) => Number(p)).filter((n) => Number.isFinite(n));
      if (nums.length) totals.push(nums[nums.length - 1]);
    }

    if (!totals.length) {
      parsedOutline.value = { meta, stats: { min: 0, max: 0, avg: 0, p50: 0, p90: 0, p99: 0 }, totals: [] };
      if (!opts.silent) ElMessage.warning("outline.csv 未解析到数值行");
      return;
    }

    const sorted = [...totals].sort((a, b) => a - b);
    const sum = totals.reduce((a, b) => a + b, 0);

    const stats = {
      min: Number(sorted[0].toFixed(6)),
      max: Number(sorted[sorted.length - 1].toFixed(6)),
      avg: Number((sum / totals.length).toFixed(6)),
      p50: Number(quantile(sorted, 0.5).toFixed(6)),
      p90: Number(quantile(sorted, 0.9).toFixed(6)),
      p99: Number(quantile(sorted, 0.99).toFixed(6)),
    };

    parsedOutline.value = { meta, stats, totals };
    await nextTick();
    renderOutlineChart(sorted, stats);

    if (!opts.silent) ElMessage.success("outline.csv 解析完成");
  } catch (e) {
    if (!opts.silent) ElMessage.error("解析 outline.csv 失败：" + (e?.message || "未知错误"));
  }
}

function renderOutlineChart(sortedTotals, stats) {
  const el = document.getElementById("outline-chart");
  if (!el) return;

  if (el.clientWidth === 0 || el.clientHeight === 0) {
    setTimeout(() => renderOutlineChart(sortedTotals, stats), 100);
    return;
  }

  if (!outlineChart) {
    outlineChart = echarts.init(el);
    window.addEventListener("resize", () => outlineChart?.resize());
  }

  outlineChart.setOption({
    tooltip: { trigger: "axis" },
    grid: { left: "3%", right: "4%", bottom: "3%", containLabel: true },
    xAxis: { type: "category", data: sortedTotals.map((_, i) => String(i)), name: "样本序号" },
    yAxis: { type: "value", name: "总通信时间" },
    series: [{
      name: "total_comm_time",
      type: "line",
      smooth: true,
      showSymbol: false,
      data: sortedTotals,
      markLine: { data: [{ yAxis: stats.avg, name: "avg" }, { yAxis: stats.p90, name: "p90" }, { yAxis: stats.p99, name: "p99" }] },
    }],
  });
}

let outlineChart = null;

function resetForm() {
  Object.assign(commForm, initForm());
  generatedScript.value = "";
  commFormRef.value?.clearValidate?.();
  outputLog.value = "";
  currentTaskId.value = null;
  currentLogFilePath.value = "";
  currentResultDir.value = "";
  resultFiles.value = [];
  parsedOutline.value = null;
  stopStatusPolling();
  isRunning.value = false;
  ElMessage.info("表单已重置");
}

onBeforeUnmount(() => {
  stopStatusPolling();
  if (outlineChart) {
    try { outlineChart.dispose(); } catch {}
  }
});
</script>

<style scoped>
.comm-collection-content { padding: 20px; }
.params-card { width: 100%; }
.param-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
.param-row-actions { display: flex; gap: 10px; margin-top: 8px; }
.script-container { max-height: 600px; overflow-y: auto; background-color: #1e1e1e; padding: 15px; border-radius: 4px; }
.generated-script { margin: 0; color: #d4d4d4; font-family: "Courier New", monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-wrap: break-word; }
.log-card { margin-top: 20px; background-color: #f5f7fa; }
.log-output { max-height: 400px; overflow-y: auto; padding: 10px; font-size: 12px; line-height: 1.5; color: #333; white-space: pre-wrap; word-wrap: break-word; }
</style>
