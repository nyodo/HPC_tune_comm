<template>
  <div class="collection-content">
    <el-card>
      <h4>I/O数据采集</h4>

      <el-form
        ref="ioFormRef"
        :model="ioForm"
        :rules="ioRules"
        label-width="200px"
        size="small"
        status-icon
      >
        <!-- 基本配置 -->
        <el-divider content-position="left">基本配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="作业名称" prop="name">
              <el-input v-model="ioForm.name" placeholder="请输入作业名称" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="结果文件夹" prop="result_folder">
              <el-input v-model="ioForm.result_folder" placeholder="请输入结果文件夹路径" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="节点数" prop="nodes">
              <el-input-number v-model="ioForm.nodes" :min="1" :precision="0" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="MPI模式" prop="MPI_mode">
              <el-select v-model="ioForm.MPI_mode" placeholder="请选择MPI模式">
                <el-option label="OpenMPI" value="openmpi" />
                <el-option label="POSIX" value="posixio" />
                <el-option label="MPICH" value="mpich" />
              </el-select>
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="每节点进程数" prop="tasks_per_node">
              <el-input-number
                v-model="ioForm.tasks_per_node"
                :min="1"
                :max="16"
                :precision="0"
                placeholder="1~16"
              />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="运行命令" prop="run_command">
              <el-input v-model="ioForm.run_command" placeholder="请输入运行命令" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="模型名称" prop="model_name">
              <el-input v-model="ioForm.model_name" placeholder="请输入模型名称" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="加载模块" prop="module_load">
              <el-switch v-model="ioForm.module_load" active-text="是" inactive-text="否" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="自定义模块">
              <el-input v-model="ioForm.self_module" placeholder="可空" clearable />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="自定义导出">
              <el-input v-model="ioForm.self_export" placeholder="可空" clearable />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- 训练配置 -->
        <el-divider content-position="left">训练配置</el-divider>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="迭代起始" prop="iter_start">
              <el-input-number v-model="ioForm.iter_start" :min="0" :precision="0" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="迭代结束" prop="iter_end">
              <el-input-number v-model="ioForm.iter_end" :min="0" :precision="0" placeholder="可空" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="训练模式" prop="train_mode">
              <el-select v-model="ioForm.train_mode">
                <el-option label="新训练" value="new" />
                <el-option label="继续训练" value="continue" />
              </el-select>
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="保存轮次" prop="save_rounds">
              <el-input-number v-model="ioForm.save_rounds" :min="1" :precision="0" />
            </el-form-item>
          </el-col>

          <el-col :span="12">
            <el-form-item label="分区" prop="partition">
              <el-input v-model="ioForm.partition" placeholder="请输入分区名称" clearable />
            </el-form-item>
          </el-col>
        </el-row>

        <!-- Slurm Shell配置 -->
        <el-divider content-position="left">Slurm Shell配置</el-divider>
        <el-form-item label="Shell文件路径" prop="slurm_shell.file_path">
          <el-input v-model="ioForm.slurm_shell.file_path" placeholder="例如：ior_ompi.sh（可空）" clearable />
        </el-form-item>

        <!-- OpenMPI 参数（真实应用：列表输入，支持 1k/1m/1g/1t） -->
        <el-form-item v-if="ioForm.MPI_mode === 'openmpi'" label="OpenMPI参数">
          <el-card shadow="hover" class="params-card">
            <el-form-item
              label="分组选项（列表）"
              prop="slurm_shell.parameters.__OMPI_MCA_io_ompio_grouping_option"
            >
              <el-input
                v-model="ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_grouping_option"
                placeholder="例如：1,2,3,4,5,6,7"
                clearable
                @input="filterListIntText('__OMPI_MCA_io_ompio_grouping_option', false)"
              />
            </el-form-item>

            <el-form-item
              label="聚合器数量（列表）"
              prop="slurm_shell.parameters.__OMPI_MCA_io_ompio_num_aggregators"
            >
              <el-input
                v-model="ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_num_aggregators"
                placeholder="例如：2,8,32（允许 -1）"
                clearable
                @input="filterListIntText('__OMPI_MCA_io_ompio_num_aggregators', true)"
              />
            </el-form-item>

            <el-form-item
              label="循环缓冲区大小（列表）"
              prop="slurm_shell.parameters.__OMPI_MCA_io_ompio_cycle_buffer_size"
            >
              <el-input
                v-model="ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_cycle_buffer_size"
                placeholder="例如：1m,32m,128m,512m（支持 k/m/g/t）"
                clearable
                @input="filterListSizeText('__OMPI_MCA_io_ompio_cycle_buffer_size')"
              />
            </el-form-item>

            <el-form-item
              label="聚合器字节数（列表）"
              prop="slurm_shell.parameters.__OMPI_MCA_io_ompio_bytes_per_agg"
            >
              <el-input
                v-model="ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_bytes_per_agg"
                placeholder="例如：1m,32m,64m（支持 k/m/g/t）"
                clearable
                @input="filterListSizeText('__OMPI_MCA_io_ompio_bytes_per_agg')"
              />
            </el-form-item>

            <el-form-item
              label="blocksize, transfersize, segment"
              prop="slurm_shell.parameters.blocksize_transfersize_segment"
            >
              <el-input
                v-model="ioForm.slurm_shell.parameters.blocksize_transfersize_segment"
                placeholder="例如：2m, 2m, 1（前两项支持 k/m/g/t）"
                clearable
                @input="filterBTSText()"
              />
            </el-form-item>
          </el-card>
        </el-form-item>

        <!-- 输入文件配置：支持任意 parameters 键 & 值列表 -->
        <el-divider content-position="left">输入文件配置（生成 input_file_1..N）</el-divider>

        <div v-for="(file, index) in ioForm.input_files" :key="index" class="input-file-item">
          <div class="file-header">
            <el-divider content-position="left">input_file_{{ index + 1 }}</el-divider>
            <el-button
              v-if="ioForm.input_files.length > 1"
              type="danger"
              size="small"
              @click="removeInputFile(index)"
              class="remove-file-btn"
            >
              删除文件
            </el-button>
          </div>

          <el-form-item :label="'文件路径'" :prop="`input_files.${index}.file_path`">
            <el-input v-model="file.file_path" placeholder="例如：romio_hints / misa_config.yaml" clearable />
          </el-form-item>

          <el-form-item label="参数（键 -> 值列表）">
            <div class="param-rows">
              <div
                v-for="(row, rIdx) in file.param_rows"
                :key="rIdx"
                class="param-row"
              >
                <el-input
                  v-model="row.key"
                  class="param-key"
                  placeholder="参数名，例如：__striping_unit 或 __X__"
                  clearable
                />
                <el-input
                  v-model="row.valuesText"
                  class="param-values"
                  placeholder="值列表：支持逗号/空格/换行；支持 1k/1m/1g/1t，例如：512k,1m,2m"
                  clearable
                />
                <el-button
                  type="danger"
                  plain
                  size="small"
                  @click="removeParamRow(index, rIdx)"
                >
                  删除
                </el-button>
              </div>

              <div class="param-row-actions">
                <el-button size="small" @click="addParamRow(index)">添加参数行</el-button>
                <el-button size="small" type="warning" plain @click="clearParamRows(index)">清空参数</el-button>
              </div>

              <div class="param-hint">
                说明：提交时会把每行的值解析成 JSON 数组字符串，例如 1m → 1048576；空行会自动忽略。
              </div>
            </div>
          </el-form-item>
        </div>

        <el-form-item>
          <el-button type="primary" size="small" @click="addInputFile">添加输入文件</el-button>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="startCollection">开始采集</el-button>
          <el-button @click="resetIOForm">重置</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, watch } from "vue";
import { ElMessage } from "element-plus";
import axios from "axios";

const ioFormRef = ref(null);

function newInputFile() {
  return {
    file_path: "",
    // 这里用 param_rows 承载任意参数（键+列表文本）
    param_rows: [
      { key: "", valuesText: "" },
    ],
  };
}

const ioForm = reactive({
  name: "",
  result_folder: "",
  nodes: 8,
  MPI_mode: "openmpi", // openmpi | posixio | mpich
  tasks_per_node: 16,
  run_command: "",
  model_name: "",
  self_module: "",
  self_export: "",
  module_load: true,
  iter_start: 0,
  iter_end: null,
  train_mode: "new",
  save_rounds: 10,
  partition: "",
  slurm_shell: {
    file_path: "",
    parameters: {
      __OMPI_MCA_io_ompio_grouping_option: "1,2,3,4,5,6,7",
      __OMPI_MCA_io_ompio_num_aggregators: "2,8,32",
      __OMPI_MCA_io_ompio_cycle_buffer_size: "1m,32m,128m,512m",
      __OMPI_MCA_io_ompio_bytes_per_agg: "1m,32m,64m",
      blocksize_transfersize_segment: "2m, 2m, 1",
    },
  },
  input_files: [newInputFile()],
});

/* ---------------- 解析：1k/1m/1g/1t ---------------- */
function parseValueToken(token) {
  const raw = String(token ?? "").trim();
  if (!raw) return null;

  // 先尝试纯整数（允许负号）
  if (/^-?\d+$/.test(raw)) return String(parseInt(raw, 10));

  // 再尝试 size：1k/1m/1g/1t（允许负号）
  const m = raw.match(/^(-?\d+)\s*([kKmMgGtT])$/);
  if (m) {
    const base = parseInt(m[1], 10);
    const unit = m[2].toLowerCase();
    const mul =
      unit === "k" ? 1024 :
      unit === "m" ? 1024 ** 2 :
      unit === "g" ? 1024 ** 3 :
      1024 ** 4; // t
    return String(base * mul);
  }

  // 其他字符串原样保留（例如 auto / on / romio_hints 等）
  return raw;
}

function splitTokens(text) {
  const s = String(text ?? "").trim();
  if (!s) return [];
  return s.split(/[,;\n\s]+/g).map((x) => x.trim()).filter(Boolean);
}

function parseValuesToArray(valuesText) {
  const toks = splitTokens(valuesText);
  const out = [];
  for (const t of toks) {
    const v = parseValueToken(t);
    if (v !== null) out.push(v);
  }
  return out;
}

/* ---------------- OpenMPI：输入框过滤（仅 UI 体验，不影响最终解析） ---------------- */
function filterListIntText(fieldKey, allowNegative = false) {
  const v = String(ioForm.slurm_shell.parameters[fieldKey] ?? "");
  const re = allowNegative ? /[^\d,\s-]/g : /[^\d,\s]/g;
  ioForm.slurm_shell.parameters[fieldKey] = v.replace(re, "").replace(/(?!^)-/g, "");
}
function filterListSizeText(fieldKey) {
  const v = String(ioForm.slurm_shell.parameters[fieldKey] ?? "");
  ioForm.slurm_shell.parameters[fieldKey] = v.replace(/[^0-9kmgtKMGT,\s-]/g, "");
}
function filterBTSText() {
  const v = String(ioForm.slurm_shell.parameters.blocksize_transfersize_segment ?? "");
  ioForm.slurm_shell.parameters.blocksize_transfersize_segment = v.replace(/[^0-9kmgtKMGT,\s-]/g, "");
}

/* ---------------- 校验 ---------------- */
const rangeNumber = (min, max, label = "该项") => (rule, value, callback) => {
  if (value === null || value === undefined || value === "") return callback(new Error(`${label}不能为空`));
  const n = Number(value);
  if (!Number.isFinite(n)) return callback(new Error(`${label}必须为数字`));
  if (n < min || n > max) return callback(new Error(`${label}范围：${min}~${max}`));
  return callback();
};

const iterEndValidator = (rule, value, callback) => {
  if (value === null || value === undefined || value === "") return callback(); // 可空
  const start = Number(ioForm.iter_start);
  const end = Number(value);
  if (Number.isNaN(end)) return callback(new Error("迭代结束必须为数字"));
  if (end < start) return callback(new Error("迭代结束必须 >= 迭代起始"));
  return callback();
};

function openmpiListValidator(kind, allowNegative = false) {
  return (rule, value, callback) => {
    if (ioForm.MPI_mode !== "openmpi") return callback();
    const v = String(value ?? "").trim();
    if (!v) return callback(new Error("该项为必填"));

    if (kind === "int") {
      for (const t of splitTokens(v)) {
        if (!(allowNegative ? /^-?\d+$/.test(t) : /^\d+$/.test(t))) {
          return callback(new Error("格式错误：请输入整数列表（可逗号/空格分隔）"));
        }
      }
      return callback();
    }

    if (kind === "size") {
      for (const t of splitTokens(v)) {
        const ok = /^-?\d+([kKmMgGtT])?$/.test(t);
        if (!ok) return callback(new Error("格式错误：支持 1k/1m/1g/1t 或纯整数"));
      }
      return callback();
    }

    if (kind === "bts") {
      const parts = v.split(",").map((x) => x.trim()).filter(Boolean);
      if (parts.length !== 3) return callback(new Error("格式错误：必须为 3 段，例如：2m, 2m, 1"));
      if (!/^-?\d+([kKmMgGtT])?$/.test(parts[0])) return callback(new Error("blocksize 格式错误"));
      if (!/^-?\d+([kKmMgGtT])?$/.test(parts[1])) return callback(new Error("transfersize 格式错误"));
      if (!/^\d+$/.test(parts[2])) return callback(new Error("segment 必须为整数"));
      return callback();
    }

    return callback();
  };
}

function validateInputFilesBasic() {
  if (!Array.isArray(ioForm.input_files) || ioForm.input_files.length < 1) {
    ElMessage.error("至少需要一个输入文件");
    return false;
  }
  for (let i = 0; i < ioForm.input_files.length; i++) {
    const fp = (ioForm.input_files[i]?.file_path || "").trim();
    if (!fp) {
      ElMessage.error(`请输入 input_file_${i + 1} 的文件路径`);
      return false;
    }
  }
  return true;
}

const ioRules = reactive({
  name: [{ required: true, message: "请输入作业名称", trigger: "blur" }],
  result_folder: [{ required: true, message: "请输入结果文件夹路径", trigger: "blur" }],
  nodes: [{ required: true, type: "number", message: "请输入节点数", trigger: "change" }],
  MPI_mode: [{ required: true, message: "请选择MPI模式", trigger: "change" }],
  tasks_per_node: [
    { required: true, type: "number", message: "请输入每节点进程数", trigger: "change" },
    { validator: rangeNumber(1, 16, "每节点进程数"), trigger: ["change", "blur"] },
  ],
  run_command: [{ required: true, message: "请输入运行命令", trigger: "blur" }],
  model_name: [{ required: true, message: "请输入模型名称", trigger: "blur" }],
  iter_start: [{ required: true, type: "number", message: "请输入迭代起始值", trigger: "change" }],
  iter_end: [{ validator: iterEndValidator, trigger: ["change", "blur"] }],
  train_mode: [{ required: true, message: "请选择训练模式", trigger: "change" }],
  save_rounds: [{ required: true, type: "number", message: "请输入保存轮次", trigger: "change" }],
  partition: [{ required: true, message: "请输入分区名称", trigger: "blur" }],
  "slurm_shell.file_path": [{ message: "Shell文件路径可为空", trigger: "blur" }],

  // OpenMPI 参数校验
  "slurm_shell.parameters.__OMPI_MCA_io_ompio_grouping_option": [
    { validator: openmpiListValidator("int", false), trigger: ["blur", "change"] },
  ],
  "slurm_shell.parameters.__OMPI_MCA_io_ompio_num_aggregators": [
    { validator: openmpiListValidator("int", true), trigger: ["blur", "change"] },
  ],
  "slurm_shell.parameters.__OMPI_MCA_io_ompio_cycle_buffer_size": [
    { validator: openmpiListValidator("size", false), trigger: ["blur", "change"] },
  ],
  "slurm_shell.parameters.__OMPI_MCA_io_ompio_bytes_per_agg": [
    { validator: openmpiListValidator("size", false), trigger: ["blur", "change"] },
  ],
  "slurm_shell.parameters.blocksize_transfersize_segment": [
    { validator: openmpiListValidator("bts", false), trigger: ["blur", "change"] },
  ],
});

/* ---------------- input_file 动态参数行操作 ---------------- */
function addParamRow(fileIndex) {
  ioForm.input_files[fileIndex].param_rows.push({ key: "", valuesText: "" });
}
function removeParamRow(fileIndex, rowIndex) {
  const rows = ioForm.input_files[fileIndex].param_rows;
  if (rows.length <= 1) {
    rows[0].key = "";
    rows[0].valuesText = "";
    return;
  }
  rows.splice(rowIndex, 1);
}
function clearParamRows(fileIndex) {
  ioForm.input_files[fileIndex].param_rows = [{ key: "", valuesText: "" }];
}

/* ---------------- 构建 parameters：任意 key -> string[] ---------------- */
function buildParametersFromRows(param_rows) {
  const parameters = {};
  for (const row of param_rows || []) {
    const key = String(row.key || "").trim();
    if (!key) continue;

    const arr = parseValuesToArray(row.valuesText);
    if (arr.length === 0) continue;

    parameters[key] = arr;
  }
  return parameters;
}

/* ---------------- 规范化提交：生成 input_file_1..N 顶层格式 ---------------- */
function normalizeIoPayload() {
  const toIntOrNull = (v) => {
    if (v === null || v === undefined || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : null;
  };
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  const payload = {
    name: ioForm.name,
    result_folder: ioForm.result_folder,
    nodes: toIntOrNull(ioForm.nodes),
    MPI_mode: ioForm.MPI_mode,
    tasks_per_node: clamp(toIntOrNull(ioForm.tasks_per_node) ?? 1, 1, 16),
    run_command: ioForm.run_command,
    model_name: ioForm.model_name,

    iter_start: toIntOrNull(ioForm.iter_start),
    iter_end: ioForm.iter_end === null || ioForm.iter_end === "" ? null : toIntOrNull(ioForm.iter_end),

    train_mode: ioForm.train_mode,
    save_rounds: toIntOrNull(ioForm.save_rounds),
    partition: ioForm.partition,

    self_module: ioForm.self_module ? ioForm.self_module : null,
    self_export: ioForm.self_export ? ioForm.self_export : null,
    module_load: !!ioForm.module_load,

    slurm_shell: {
      file_path: ioForm.slurm_shell?.file_path ?? "",
      parameters: {},
    },
  };

  // slurm_shell.parameters
  if (payload.MPI_mode === "openmpi") {
    const p = ioForm.slurm_shell?.parameters || {};

    const grouping = parseValuesToArray(p.__OMPI_MCA_io_ompio_grouping_option);
    const aggs = parseValuesToArray(p.__OMPI_MCA_io_ompio_num_aggregators);
    const cycle = parseValuesToArray(p.__OMPI_MCA_io_ompio_cycle_buffer_size);
    const bytes = parseValuesToArray(p.__OMPI_MCA_io_ompio_bytes_per_agg);

    // bts：输出 ["<blocksize>, <transfersize>, <segment>"] 一条字符串，匹配你示例
    const btsParts = String(p.blocksize_transfersize_segment || "")
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    if (btsParts.length !== 3) {
      throw new Error("blocksize, transfersize, segment 必须为 3 段，例如：2m, 2m, 1");
    }
    const b0 = parseValueToken(btsParts[0]);
    const b1 = parseValueToken(btsParts[1]);
    const b2 = /^\d+$/.test(btsParts[2]) ? String(parseInt(btsParts[2], 10)) : null;
    if (b0 === null || b1 === null || b2 === null) {
      throw new Error("blocksize, transfersize, segment 解析失败，请检查输入");
    }
    const bts = [`${b0}, ${b1}, ${b2}`];

    if (!grouping.length || !aggs.length || !cycle.length || !bytes.length) {
      throw new Error("OpenMPI 参数不能为空且必须为合法列表");
    }

    payload.slurm_shell.parameters = {
      "__OMPI_MCA_io_ompio_grouping_option": grouping,
      "__OMPI_MCA_io_ompio_num_aggregators": aggs,
      "__OMPI_MCA_io_ompio_cycle_buffer_size": cycle,
      "__OMPI_MCA_io_ompio_bytes_per_agg": bytes,
      "blocksize, transfersize, segment": bts,
    };
  } else {
    // posixio / mpich：必须是 {}
    payload.slurm_shell.parameters = {};
  }

  // input_file_1..N 顶层键
  (ioForm.input_files || []).forEach((f, idx) => {
    payload[`input_file_${idx + 1}`] = {
      file_path: f.file_path,
      parameters: buildParametersFromRows(f.param_rows),
    };
  });

  return payload;
}

/* ---------------- 事件 ---------------- */
const startCollection = async () => {
  try {
    const ok = await ioFormRef.value?.validate?.();
    if (!ok) return;

    if (!validateInputFilesBasic()) return;

    const payload = normalizeIoPayload();

    // 告诉后端执行哪一步：1 = 数据采集
    payload.step = "1";

    await axios.post("/api/io-collection", payload, {
      headers: { "Content-Type": "application/json" },
    });

    ElMessage.success("表单提交成功，开始I/O数据采集");
  } catch (error) {
    console.error("表单提交失败:", error);
    const msg = error?.response?.data?.message || error?.message || "未知错误";
    ElMessage.error(`表单提交失败：${msg}`);
  }
};

const resetIOForm = () => {
  ioForm.name = "";
  ioForm.result_folder = "";
  ioForm.nodes = 8;
  ioForm.MPI_mode = "openmpi";
  ioForm.tasks_per_node = 16;
  ioForm.run_command = "";
  ioForm.model_name = "";
  ioForm.self_module = "";
  ioForm.self_export = "";
  ioForm.module_load = true;
  ioForm.iter_start = 0;
  ioForm.iter_end = null;
  ioForm.train_mode = "new";
  ioForm.save_rounds = 10;
  ioForm.partition = "";

  ioForm.slurm_shell.file_path = "";
  ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_grouping_option = "1,2,3,4,5,6,7";
  ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_num_aggregators = "2,8,32";
  ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_cycle_buffer_size = "1m,32m,128m,512m";
  ioForm.slurm_shell.parameters.__OMPI_MCA_io_ompio_bytes_per_agg = "1m,32m,64m";
  ioForm.slurm_shell.parameters.blocksize_transfersize_segment = "2m, 2m, 1";

  ioForm.input_files = [newInputFile()];

  ioFormRef.value?.clearValidate?.();
  ElMessage.info("已重置I/O数据采集表单");
};

const addInputFile = () => {
  ioForm.input_files.push(newInputFile());
};

const removeInputFile = (index) => {
  if (ioForm.input_files.length <= 1) {
    ElMessage.warning("至少需要保留一个输入文件");
    return;
  }
  ioForm.input_files.splice(index, 1);
};

// 切换 MPI：非 openmpi 时，清空 OpenMPI 校验提示（payload 会自动 parameters={}）
watch(
  () => ioForm.MPI_mode,
  (mode) => {
    if (mode !== "openmpi") {
      ioFormRef.value?.clearValidate?.([
        "slurm_shell.parameters.__OMPI_MCA_io_ompio_grouping_option",
        "slurm_shell.parameters.__OMPI_MCA_io_ompio_num_aggregators",
        "slurm_shell.parameters.__OMPI_MCA_io_ompio_cycle_buffer_size",
        "slurm_shell.parameters.__OMPI_MCA_io_ompio_bytes_per_agg",
        "slurm_shell.parameters.blocksize_transfersize_segment",
      ]);
    }
  }
);
</script>

<style scoped>
.collection-content {
  padding: 20px;
}
.input-file-item {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.file-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}
.remove-file-btn {
  margin-top: -10px;
}
.params-card {
  width: 100%;
}

.param-rows {
  width: 100%;
}
.param-row {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 10px;
}
.param-key {
  width: 260px;
}
.param-values {
  flex: 1;
}
.param-row-actions {
  display: flex;
  gap: 10px;
  margin-top: 8px;
}
.param-hint {
  margin-top: 8px;
  font-size: 12px;
  color: #666;
}
</style>
