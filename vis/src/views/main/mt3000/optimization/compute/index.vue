<template>
  <div class="optimization-content">
    <el-form :model="form" label-width="140px">
      <el-divider content-position="left">Kernel</el-divider>

      <el-form-item label="name" required>
        <el-input v-model="form.kernel.name" placeholder="例如：vector_add" />
      </el-form-item>

      <el-form-item label="path" required>
        <el-input v-model="form.kernel.path" placeholder="例如：kernel.c（必须是真实路径）" />
      </el-form-item>

      <el-form-item label="lang">
        <el-input v-model="form.kernel.lang" placeholder="例如：Hthreads" />
      </el-form-item>

      <el-form-item label="compiler_options">
        <el-select v-model="form.kernel.compiler_options" multiple filterable allow-create default-first-option placeholder="例如：-O2">
          <el-option
            v-for="opt in form.kernel.compiler_options"
            :key="opt"
            :label="opt"
            :value="opt"
          />
        </el-select>
      </el-form-item>

      <el-form-item label="problem_size">
        <el-input-number v-model="form.kernel.problem_size" :min="1" :step="1" />
      </el-form-item>

      <el-divider content-position="left">Inputs</el-divider>

      <el-form-item>
        <el-button type="primary" @click="addScalarInput">新增标量输入</el-button>
        <el-button type="primary" @click="addArrayInput">新增数组输入</el-button>
      </el-form-item>

      <div v-for="(it, idx) in form.inputs" :key="it.id" class="input-card">
        <el-card shadow="never">
          <template #header>
            <div class="card-header">
              <span>Input #{{ idx + 1 }}</span>
              <div>
                <el-tag size="small" type="info" v-if="it.kind === 'scalar'">scalar</el-tag>
                <el-tag size="small" type="success" v-else>array</el-tag>
                <el-button link type="danger" @click="removeInput(idx)">删除</el-button>
              </div>
            </div>
          </template>

          <el-form-item label="name" required>
            <el-input v-model="it.name" placeholder="例如：n / A / B / C" />
          </el-form-item>

          <el-form-item label="type" required>
            <el-select v-model="it.type" placeholder="请选择类型">
              <el-option label="int32" value="int32" />
              <el-option label="int64" value="int64" />
              <el-option label="float32" value="float32" />
              <el-option label="float64" value="float64" />
            </el-select>
          </el-form-item>

          <template v-if="it.kind === 'scalar'">
            <el-form-item label="value" required>
              <el-input v-model="it.value" placeholder="例如：100000" />
            </el-form-item>
          </template>

          <template v-else>
            <el-form-item label="shape" required>
              <el-input v-model="it.shapeText" placeholder="例如：100000 或 256,256" />
            </el-form-item>

            <el-form-item label="source.type" required>
              <el-select v-model="it.source.type" placeholder="请选择初始化方式">
                <el-option label="random" value="random" />
                <el-option label="zero" value="zero" />
              </el-select>
            </el-form-item>

            <template v-if="it.source.type === 'random'">
              <el-form-item label="distribution">
                <el-select v-model="it.source.distribution" placeholder="例如：normal">
                  <el-option label="normal" value="normal" />
                  <el-option label="uniform" value="uniform" />
                </el-select>
              </el-form-item>

              <el-form-item label="mean">
                <el-input-number v-model="it.source.params.mean" :step="0.1" />
              </el-form-item>
              <el-form-item label="stddev">
                <el-input-number v-model="it.source.params.stddev" :min="0" :step="0.1" />
              </el-form-item>
            </template>
          </template>
        </el-card>
      </div>

      <el-divider content-position="left">Tuning</el-divider>

      <el-form-item label="parameters">
        <div class="tuning-params">
          <div v-for="(p, idx) in form.tuning.parameters" :key="p.id" class="tuning-param-row">
            <el-input v-model="p.key" placeholder="参数名，例如：PARA1" style="width: 180px" />
            <el-input v-model="p.valuesText" placeholder="参数值列表，逗号分隔，例如：0,1,2" style="flex: 1" />
            <el-button link type="danger" @click="removeTuningParam(idx)">删除</el-button>
          </div>
          <el-button type="primary" plain @click="addTuningParam">新增参数</el-button>
        </div>
      </el-form-item>

      <el-form-item label="strategy">
        <el-input v-model="form.tuning.strategy" placeholder="例如：greedy_mls" />
      </el-form-item>

      <el-form-item label="modelfit.enabled">
        <el-switch v-model="form.tuning.modelfit.enabled" />
      </el-form-item>

      <el-form-item label="cache">
        <el-input v-model="form.tuning.cache" placeholder="例如：tuning_cache.json" />
      </el-form-item>

      <el-form-item label="log">
        <el-select v-model="form.tuning.log" placeholder="例如：DEBUG">
          <el-option label="DEBUG" value="DEBUG" />
          <el-option label="INFO" value="INFO" />
          <el-option label="WARN" value="WARN" />
          <el-option label="ERROR" value="ERROR" />
        </el-select>
      </el-form-item>

      <el-divider content-position="left">生成/提交</el-divider>

      <el-form-item>
        <el-button type="primary" @click="generateYaml">生成 YAML / 预览</el-button>
        <el-button type="success" :disabled="!yamlText" @click="submitToBackend">提交 YAML 到后端</el-button>
        <el-button :disabled="!yamlText" @click="downloadYaml">下载 config.yaml</el-button>
      </el-form-item>

      <el-form-item label="YAML 预览">
        <el-input v-model="yamlText" type="textarea" :rows="18" placeholder="点击生成后会在此显示 YAML" />
      </el-form-item>

      <el-divider content-position="left">Slurm</el-divider>

      <el-form-item label="#SBATCH -J" required>
        <el-input v-model="form.slurm.jobName" placeholder="例如：adtuner" />
      </el-form-item>

      <el-form-item label="#SBATCH -N" required>
        <el-input-number v-model="form.slurm.nodes" :min="1" :step="1" />
      </el-form-item>

      <el-form-item label="#SBATCH -n" required>
        <el-input-number v-model="form.slurm.ntasks" :min="1" :step="1" />
      </el-form-item>

      <el-form-item label="#SBATCH -p" required>
        <el-input v-model="form.slurm.partition" placeholder="例如：thmt1" />
      </el-form-item>

      <el-form-item label="#SBATCH -o" required>
        <el-input v-model="form.slurm.output" placeholder="例如：result.out" />
      </el-form-item>

      <el-form-item label="conda activate">
        <el-input v-model="form.slurm.condaActivate" placeholder="例如：/thfs3/home/xxx/miniconda3/bin/activate" />
      </el-form-item>

      <el-form-item label="conda env">
        <el-input v-model="form.slurm.condaEnv" placeholder="例如：tune" />
      </el-form-item>

      <el-form-item label="rm cache">
        <el-switch v-model="form.slurm.cleanupCache" />
      </el-form-item>

      <el-form-item v-if="form.slurm.cleanupCache" label="cache file">
        <el-input v-model="form.slurm.cacheFile" placeholder="例如：tuning_cache.json" />
      </el-form-item>

      <el-form-item label="command" required>
        <el-input v-model="form.slurm.command" placeholder="例如：adtuner config.yaml" />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" @click="generateSlurm">生成 Slurm / 预览</el-button>
        <el-button type="success" :disabled="!slurmText" @click="submitSlurmToBackend">提交 Slurm 到后端</el-button>
        <el-button :disabled="!slurmText" @click="downloadSlurm">下载 slurm.sh</el-button>
      </el-form-item>

      <el-form-item label="Slurm 预览">
        <el-input v-model="slurmText" type="textarea" :rows="12" placeholder="点击生成后会在此显示 Slurm 脚本" />
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup>
import { reactive, ref } from "vue";
import { ElMessage } from "element-plus";
import axios from "axios";

const yamlText = ref("");
const slurmText = ref("");

const uid = () => `${Date.now()}_${Math.random().toString(16).slice(2)}`;

const form = reactive({
  slurm: {
    jobName: "adtuner",
    nodes: 1,
    ntasks: 1,
    partition: "thmt1",
    output: "result.out",
    condaActivate: "/thfs3/home/xjtu_cx/miniconda3/bin/activate",
    condaEnv: "tune",
    cleanupCache: true,
    cacheFile: "tuning_cache.json",
    command: "adtuner config.yaml",
  },


  kernel: {
    name: "vector_add",
    path: "kernel.c",
    lang: "Hthreads",
    compiler_options: ["-O2"],
    problem_size: 100000,
  },
  inputs: [
    { id: uid(), kind: "scalar", name: "n", type: "int32", value: "100000" },
    {
      id: uid(),
      kind: "array",
      name: "A",
      type: "float32",
      shapeText: "100000",
      source: { type: "random", distribution: "normal", params: { mean: 3.14, stddev: 0.1 } },
    },
    {
      id: uid(),
      kind: "array",
      name: "B",
      type: "float32",
      shapeText: "100000",
      source: { type: "random", distribution: "normal", params: { mean: 3.14, stddev: 0.1 } },
    },
    {
      id: uid(),
      kind: "array",
      name: "C",
      type: "float32",
      shapeText: "100000",
      source: { type: "zero" },
    },
  ],
  tuning: {
    parameters: [
      { id: uid(), key: "PARA1", valuesText: "0,1" },
      { id: uid(), key: "PARA2", valuesText: "0,1" },
      { id: uid(), key: "hthreads_num_threads", valuesText: "4,8,12,16,20,24" },
    ],
    strategy: "greedy_mls",
    modelfit: { enabled: false },
    cache: "tuning_cache.json",
    log: "DEBUG",
  },
});

const addScalarInput = () => {
  form.inputs.push({ id: uid(), kind: "scalar", name: "", type: "int32", value: "" });
};

const addArrayInput = () => {
  form.inputs.push({
    id: uid(),
    kind: "array",
    name: "",
    type: "float32",
    shapeText: "",
    source: { type: "random", distribution: "normal", params: { mean: 0, stddev: 1 } },
  });
};

const removeInput = (idx) => {
  form.inputs.splice(idx, 1);
};

const addTuningParam = () => {
  form.tuning.parameters.push({ id: uid(), key: "", valuesText: "" });
};

const removeTuningParam = (idx) => {
  form.tuning.parameters.splice(idx, 1);
};

const parseShape = (shapeText) => {
  const s = (shapeText || "").trim();
  if (!s) return [];
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => Number(x))
    .filter((n) => Number.isFinite(n) && n > 0);
};

const quoteIfNeeded = (v) => {
  if (v === null || v === undefined) return "\"\"";
  const s = String(v);
  if (s === "") return "\"\"";
  // 如果包含特殊字符/空格，做简单引号处理
  if (/[:#\n\r\t]|\s/.test(s)) return JSON.stringify(s);
  // 纯数字/布尔
  if (/^-?\d+(\.\d+)?$/.test(s)) return s;
  if (s === "true" || s === "false") return s;
  return JSON.stringify(s);
};

const yamlLine = (indent, key, value) => `${" ".repeat(indent)}${key}: ${value}`;

const buildYaml = () => {
  const lines = [];

  // kernel
  lines.push("kernel:");
  lines.push(yamlLine(2, "name", quoteIfNeeded(form.kernel.name)));
  lines.push(yamlLine(2, "path", quoteIfNeeded(form.kernel.path)));
  lines.push(yamlLine(2, "lang", quoteIfNeeded(form.kernel.lang)));
  const opts = Array.isArray(form.kernel.compiler_options) ? form.kernel.compiler_options : [];
  lines.push(`${" ".repeat(2)}compiler_options: [${opts.map((x) => quoteIfNeeded(x)).join(", ")}]`);
  lines.push(yamlLine(2, "problem_size", quoteIfNeeded(form.kernel.problem_size)));
  lines.push("");

  // inputs
  lines.push("inputs:");
  (form.inputs || []).forEach((it) => {
    lines.push(`${" ".repeat(2)}- name: ${quoteIfNeeded(it.name)}`);
    lines.push(yamlLine(4, "type", quoteIfNeeded(it.type)));

    if (it.kind === "scalar") {
      lines.push(yamlLine(4, "value", quoteIfNeeded(it.value)));
      lines.push("");
      return;
    }

    const shapeArr = parseShape(it.shapeText);
    lines.push(`${" ".repeat(4)}shape: [${shapeArr.join(", ")}]`);
    lines.push(`${" ".repeat(4)}source:`);
    lines.push(yamlLine(6, "type", quoteIfNeeded(it.source?.type || "random")));

    if ((it.source?.type || "") === "random") {
      if (it.source?.distribution) lines.push(yamlLine(6, "distribution", quoteIfNeeded(it.source.distribution)));
      const mean = it.source?.params?.mean;
      const stddev = it.source?.params?.stddev;
      lines.push(`${" ".repeat(6)}params:`);
      lines.push(yamlLine(8, "mean", quoteIfNeeded(mean)));
      lines.push(yamlLine(8, "stddev", quoteIfNeeded(stddev)));
    }

    lines.push("");
  });

  // tuning
  lines.push("tuning:");
  lines.push(`${" ".repeat(2)}parameters:`);
  (form.tuning.parameters || []).forEach((p) => {
    const key = (p.key || "").trim();
    if (!key) return;

    const values = (p.valuesText || "")
      .split(/[,\n]+/)
      .map((x) => x.trim())
      .filter(Boolean)
      .map((x) => {
        // number -> number，否则当字符串
        if (/^-?\d+(\.\d+)?$/.test(x)) return x;
        return quoteIfNeeded(x);
      });

    lines.push(`${" ".repeat(4)}${key}: [${values.join(", ")}]`);
  });
  lines.push(yamlLine(2, "strategy", quoteIfNeeded(form.tuning.strategy)));
  lines.push(`${" ".repeat(2)}modelfit:`);
  lines.push(yamlLine(4, "enabled", form.tuning.modelfit.enabled ? "true" : "false"));
  lines.push(yamlLine(2, "cache", quoteIfNeeded(form.tuning.cache)));
  lines.push(yamlLine(2, "log", quoteIfNeeded(form.tuning.log)));

  return lines.join("\n").trim() + "\n";
};

const validateRequired = () => {
  if (!String(form.kernel.name || "").trim()) {
    ElMessage.error("kernel.name 为必填");
    return false;
  }
  if (!String(form.kernel.path || "").trim()) {
    ElMessage.error("kernel.path 为必填");
    return false;
  }
  for (const it of form.inputs || []) {
    if (!String(it.name || "").trim()) {
      ElMessage.error("inputs 中存在 name 为空的项");
      return false;
    }
    if (!String(it.type || "").trim()) {
      ElMessage.error("inputs 中存在 type 为空的项");
      return false;
    }
    if (it.kind === "scalar" && String(it.value || "") === "") {
      ElMessage.error(`标量输入 ${it.name || "(未命名)"} 的 value 不能为空`);
      return false;
    }
    if (it.kind === "array") {
      const shapeArr = parseShape(it.shapeText);
      if (!shapeArr.length) {
        ElMessage.error(`数组输入 ${it.name || "(未命名)"} 的 shape 不能为空`);
        return false;
      }
      if (!String(it.source?.type || "").trim()) {
        ElMessage.error(`数组输入 ${it.name || "(未命名)"} 的 source.type 不能为空`);
        return false;
      }
    }
  }
  return true;
};

const generateYaml = () => {
  if (!validateRequired()) return;
  yamlText.value = buildYaml();
  ElMessage.success("YAML 已生成");
};

const validateSlurmRequired = () => {
  if (!String(form.slurm.jobName || "").trim()) {
    ElMessage.error("slurm jobName 为必填");
    return false;
  }
  if (!String(form.slurm.partition || "").trim()) {
    ElMessage.error("slurm partition 为必填");
    return false;
  }
  if (!String(form.slurm.output || "").trim()) {
    ElMessage.error("slurm output 为必填");
    return false;
  }
  if (!String(form.slurm.command || "").trim()) {
    ElMessage.error("slurm command 为必填");
    return false;
  }
  return true;
};

const buildSlurm = () => {
  const lines = [];
  lines.push("#!/bin/bash");
  lines.push("");
  lines.push(`#SBATCH -J ${form.slurm.jobName}`);
  lines.push(`#SBATCH -N ${form.slurm.nodes}`);
  lines.push(`#SBATCH -n ${form.slurm.ntasks} -p ${form.slurm.partition}`);
  lines.push(`#SBATCH -o ${form.slurm.output}`);
  lines.push("");

  if (String(form.slurm.condaActivate || "").trim() && String(form.slurm.condaEnv || "").trim()) {
    lines.push(`source ${form.slurm.condaActivate} ${form.slurm.condaEnv}`);
    lines.push("");
  }

  if (form.slurm.cleanupCache) {
    const cf = String(form.slurm.cacheFile || "").trim() || "tuning_cache.json";
    lines.push(`rm ${cf}`);
    lines.push("");
  }

  lines.push(String(form.slurm.command || "").trim());
  return lines.join("\n").trim() + "\n";
};

const generateSlurm = () => {
  if (!validateSlurmRequired()) return;
  slurmText.value = buildSlurm();
  ElMessage.success("Slurm 脚本已生成");
};

const submitToBackend = async () => {
  try {
    if (!yamlText.value) {
      ElMessage.error("请先生成 YAML");
      return;
    }

    const blob = new Blob([yamlText.value], { type: "text/yaml" });
    const file = new File([blob], "config.yaml", { type: "text/yaml" });
    const fd = new FormData();
    fd.append("file", file);

    const resp = await axios.post(import.meta.env.VITE_URL + "/mt3000/optimization/compute", fd);
    ElMessage.success("提交成功");
    console.log(resp?.data);
  } catch (e) {
    const status = e?.response?.status;
    const msg = e?.response?.data?.error || e?.response?.data?.message;
    ElMessage.error(`提交失败${status ? ` (${status})` : ""}${msg ? `: ${msg}` : ""}`);
    console.error(e);
  }
};

const submitSlurmToBackend = async () => {
  try {
    if (!slurmText.value) {
      ElMessage.error("请先生成 Slurm 脚本");
      return;
    }

    const blob = new Blob([slurmText.value], { type: "text/x-shellscript" });
    const file = new File([blob], "slurm.sh", { type: "text/x-shellscript" });
    const fd = new FormData();
    fd.append("file", file);

    const resp = await axios.post(import.meta.env.VITE_URL + "/mt3000/optimization/compute/slurm", fd);
    ElMessage.success("Slurm 提交成功");
    console.log(resp?.data);
  } catch (e) {
    const status = e?.response?.status;
    const msg = e?.response?.data?.error || e?.response?.data?.message;
    ElMessage.error(`Slurm 提交失败${status ? ` (${status})` : ""}${msg ? `: ${msg}` : ""}`);
    console.error(e);
  }
};

const downloadYaml = () => {
  try {
    if (!yamlText.value) {
      ElMessage.error("请先生成 YAML");
      return;
    }
    const blob = new Blob([yamlText.value], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "config.yaml";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    ElMessage.success("已下载 config.yaml");
  } catch (e) {
    ElMessage.error("下载失败");
    console.error(e);
  }
};

const downloadSlurm = () => {
  try {
    if (!slurmText.value) {
      ElMessage.error("请先生成 Slurm 脚本");
      return;
    }
    const blob = new Blob([slurmText.value], { type: "text/x-shellscript" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "slurm.sh";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    ElMessage.success("已下载 slurm.sh");
  } catch (e) {
    ElMessage.error("下载失败");
    console.error(e);
  }
};
</script>

<style scoped>
.optimization-content {
  padding: 20px;
}

.input-card {
  margin-bottom: 12px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.tuning-params {
  width: 100%;
}

.tuning-param-row {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}
</style>
