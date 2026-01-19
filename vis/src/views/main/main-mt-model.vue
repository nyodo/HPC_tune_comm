<template>
  <div class="foot-container">
    <div class="container">
      <h4>迈创平台访存性能自动建模</h4>
      <div class="form-section">
        <h3 class="title-label">上传数据</h3>
        <el-button type="primary" @click="openDrawer('code')">上传代码</el-button>
        <el-button type="primary" @click="openDrawer('ir')">上传中间表示</el-button>
        <el-button type="primary" @click="openDrawer('cfg')">上传控制流图</el-button>
        <el-button type="primary" @click="openDrawer('dynamicData')">上传动态数据</el-button>
        <div class="button-section">
          <el-button type="primary" @click="handleSubmit">训练</el-button>
          <el-button type="primary" @click="handleEvaluate">评估</el-button>
          <el-button @click="handleReset">重置</el-button>
        </div>
      </div>
      <div v-if="result.evaluate_image1" class="result-section">
        <h3 class="title-label">评估结果</h3>
        <!-- <h3 class="title-label" v-if="result.loss_image">损失下降曲线图</h3> -->
        <div v-if="result.evaluate_image1" class="image-container">
          <img :src="result.evaluate_image1" alt="分析图表" style="max-width: 80%; height: auto;">
        </div>
        <div v-if="result.evaluate_image2" class="image-container">
          <img :src="result.evaluate_image2" alt="分析图表" style="max-width: 80%; height: auto;">
        </div>
        <div v-if="result.evaluate_image3" class="image-container">
          <img :src="result.evaluate_image3" alt="分析图表" style="max-width: 80%; height: auto;">
        </div>
        <div class="result-content">
          <p></p>
        </div>
      </div>
      <div class="result-section">
        <h3 class="title-label">建模训练过程</h3>
        <div class="result-content">
          <p class="formatted-text">{{ result.model_process }}</p>
        </div>
        <h3 class="title-label" v-if="result.loss_image">损失下降曲线图</h3>
        <div v-if="result.loss_image" class="image-container">
          <img :src="result.loss_image" alt="分析图表" style="max-width: 80%; height: auto;">
        </div>
      </div>
    </div>

    <!-- 抽屉组件 for 代码 -->
    <sDrawer v-model="drawers.code.visible" title="上传代码" size="35%" :close-on-click-modal="false">
      <el-form :model="drawers.code.form" label-width="100px" class="drawer-form">
        <el-form-item label="上传类型:">
          <el-select v-model="drawers.code.uploadType" placeholder="选择上传类型">
            <el-option label="文本" value="text"></el-option>
            <el-option label="文件" value="file"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item v-if="drawers.code.uploadType === 'text'" label="代码:">
          <el-select v-model="drawers.code.selectedOption" placeholder="可选择样例或输入代码" class="select-input">
            <el-option v-for="option in options" :key="option.value" :label="option.label"
              :value="option.cppCode"></el-option>
          </el-select>
          <el-input type="textarea" v-model="drawers.code.inputText" rows="10" class="textarea-input"></el-input>
        </el-form-item>
        <el-form-item v-if="drawers.code.uploadType === 'file'" label="文件:">
          <el-upload class="upload-demo" drag :action="uploadFileUrl" :limit="1" accept=".txt,.c,.cpp"
            :on-success="handleSuccess('code')" :on-error="handleError('code')" :on-remove="handleRemove('code')"
            ref="uploadRefCode">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                单次分析限一个代码文件(.cpp,.c,.txt)
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="drawers.code.visible = false">取消</el-button>
        <el-button type="primary" @click="saveData('code')">保存</el-button>
      </template>
    </sDrawer>

    <!-- 抽屉组件 for 中间表示 -->
    <sDrawer v-model="drawers.ir.visible" title="上传中间表示" size="35%" :close-on-click-modal="false">
      <el-form :model="drawers.ir.form" label-width="100px" class="drawer-form">
        <el-form-item label="上传类型:">
          <el-select v-model="drawers.ir.uploadType" placeholder="选择上传类型">
            <el-option label="文本" value="text"></el-option>
            <el-option label="文件" value="file"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item v-if="drawers.ir.uploadType === 'text'" label="中间表示:">
          <el-input type="textarea" v-model="drawers.ir.inputText" rows="10" class="textarea-input"></el-input>
        </el-form-item>
        <el-form-item v-if="drawers.ir.uploadType === 'file'" label="文件:">
          <el-upload class="upload-demo" drag :action="uploadFileUrl" :limit="1" accept=".txt,.ll"
            :on-success="handleSuccess('ir')" :on-error="handleError('ir')" :on-remove="handleRemove('ir')"
            ref="uploadRefIr">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                单次分析限一个ll文件
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="drawers.ir.visible = false">取消</el-button>
        <el-button type="primary" @click="saveData('ir')">保存</el-button>
      </template>
    </sDrawer>

    <!-- 抽屉组件 for 控制流图 -->
    <sDrawer v-model="drawers.cfg.visible" title="上传控制流图" size="35%" :close-on-click-modal="false">
      <el-form :model="drawers.cfg.form" label-width="100px" class="drawer-form">
        <el-form-item label="上传类型:">
          <el-select v-model="drawers.cfg.uploadType" placeholder="选择上传类型">
            <el-option label="文本" value="text"></el-option>
            <el-option label="文件" value="file"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item v-if="drawers.cfg.uploadType === 'text'" label="控制流图:">
          <el-input type="textarea" v-model="drawers.cfg.inputText" rows="10" class="textarea-input"></el-input>
        </el-form-item>
        <el-form-item v-if="drawers.cfg.uploadType === 'file'" label="文件:">
          <el-upload class="upload-demo" drag :action="uploadFileUrl" :limit="1" accept=".txt,.dot"
            :on-success="handleSuccess('cfg')" :on-error="handleError('cfg')" :on-remove="handleRemove('cfg')"
            ref="uploadRefCfg">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                单次分析限一个dot文件
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="drawers.cfg.visible = false">取消</el-button>
        <el-button type="primary" @click="saveData('cfg')">保存</el-button>
      </template>
    </sDrawer>

    <!-- 抽屉组件 for 动态数据 -->
    <sDrawer v-model="drawers.dynamicData.visible" title="上传动态数据" size="35%" :close-on-click-modal="false">
      <el-form :model="drawers.dynamicData.form" label-width="100px" class="drawer-form">
        <el-form-item label="上传类型:">
          <el-select v-model="drawers.dynamicData.uploadType" placeholder="选择上传类型">
            <!-- <el-option label="文本" value="text"></el-option> -->
            <el-option label="文件" value="file"></el-option>
          </el-select>
        </el-form-item>
        <el-form-item label="文件:">
          <el-upload class="upload-demo" drag :action="uploadFileUrl" :limit="1" accept=".csv"
            :on-success="handleSuccess('dynamicData')" :on-error="handleError('dynamicData')"
            :on-remove="handleRemove('dynamicData')" ref="uploadRefDynamicData">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                单次分析限一个CSV文件
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="drawers.dynamicData.visible = false">取消</el-button>
        <el-button type="primary" @click="saveData('dynamicData')">保存</el-button>
      </template>
    </sDrawer>

    <!-- 页脚 -->
    <div id="footer">
      <div class="copyright"><span> Copyright© 2025 <a href="https://se.xjtu.edu.cn/" target="_blank">XJTU</a> All
          Rights
          Reserved</span></div>
      <div class="credits"></div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, watch, onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import axios from 'axios';
import MarkdownIt from 'markdown-it';
import sDrawer from "@/components/s-drawer/s-drawer.vue";

// 处理Markdown字符串
const md = new MarkdownIt();
const renderMarkdown = (text) => {
  return md.render(text);
};

// 定义所有上传组件的 refs
const uploadRefCode = ref(null);
const uploadRefIr = ref(null);
const uploadRefCfg = ref(null);
const uploadRefDynamicData = ref(null);

const uploadFileUrl = ref(import.meta.env.VITE_URL + '/upload-file');
const result = ref({
  model_process: '',
  loss_image: '',
  evaluate_image1: '',
  evaluate_image2: '',
  evaluate_image3: '',
});

const drawers = reactive({
  code: {
    visible: false,
    uploadType: 'text',
    inputText: '',
    selectedOption: '',
    files: [],
    form: {}
  },
  ir: {
    visible: false,
    uploadType: 'text',
    inputText: '',
    files: [],
    form: {}
  },
  cfg: {
    visible: false,
    uploadType: 'text',
    inputText: '',
    files: [],
    form: {}
  },
  dynamicData: {
    visible: false,
    files: [],
    form: {}
  }
});

const options = ref([
  {
    value: 'option1',
    label: '样例1：XBLUE应用热点核函数-反向高斯算法',
    cppCode: '__global__ void gauss_all_seidel_backfor(int mne, int nv, int* nc, double* a_ae, double* f,\n                                         int* ne, double* ap, double* con, double* ff)\n{\n    int i = blockIdx.x * blockDim.x + threadIdx.x;\n    if (i < mne)\n    {\n        double tmp_b = 0.0;\n        int j;\n        for (j = nc[i]; j <= nc[i + 1] - 1; j++)\n        {\n            tmp_b += a_ae[j] * f[(nv - 1) * mne + ne[j] - 1];\n        }\n        ff[i] = (tmp_b + con[i]) / ap[i];\n    }\n}'
  },
  {
    value: 'option2',
    label: '样例2：BICG核函数',
    cppCode: '__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_NY){\n\t\ts[j] = 0.0f;\n\t\tint i;\n\t\tfor(i = 0; i < _PB_NX; i++){\n\t\t\ts[j] += r[i] * A[i * NY + j];\n\t\t}}}\n\n__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q){\n\tint i = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (i < _PB_NX){\n\t\tq[i] = 0.0f;\n\t\tint j;\n\t\tfor(j=0; j < _PB_NY; j++){\n\t\t\tq[i] += A[i * NY + j] * p[j];\n\t\t}}}'
  },
  {
    value: 'option3',
    label: '样例3：CORRELATION核函数',
    cppCode: '__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_M){\n\t\tmean[j] = 0.0;\n\t\tint i;\n\t\tfor(i=0; i < _PB_N; i++){\n\t\t\tmean[j] += data[i*M + j];\n\t\t}\n\t\tmean[j] /= (DATA_TYPE)FLOAT_N;}}\n\n__global__ void std_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_M){\n\t\tstd[j] = 0.0;\n\t\tint i;\n\t\tfor(i = 0; i < _PB_N; i++){\n\t\t\tstd[j] += (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j]);\n\t\t}\n\t\tstd[j] /= (FLOAT_N);\n\t\tstd[j] = sqrt(std[j]);\n\t\tif(std[j] <= EPS){\n\t\t\tstd[j] = 1.0;}}}\n\n__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i = blockIdx.y * blockDim.y + threadIdx.y;\n\tif ((i < _PB_N) && (j < _PB_M)){\n\t\tdata[i*M + j] -= mean[j];\n\t\tdata[i*M + j] /= (sqrt(FLOAT_N) * std[j]);}}\n\n__global__ void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data){\n\tint j1 = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i, j2;\n\tif (j1 < (_PB_M-1)){\n\t\tsymmat[j1*M + j1] = 1.0;\n\t\tfor (j2 = (j1 + 1); j2 < _PB_M; j2++){\n\t\t\tsymmat[j1*M + j2] = 0.0;\n\t\t\tfor(i = 0; i < _PB_N; i++){\n\t\t\t\tsymmat[j1*M + j2] += data[i*M + j1] * data[i*M + j2];\n\t\t\t}\n\t\t\tsymmat[j2*M + j1] = symmat[j1*M + j2];}}}'
  }
]);

watch(() => drawers.code.selectedOption, (newValue) => {
  if (newValue) {
    drawers.code.inputText = newValue;
  }
});

const openDrawer = (type) => {
  drawers[type].visible = true;
};

const saveData = (type) => {
  const drawer = drawers[type];
  if (type === 'dynamicData') {
    drawer.form = { type: 'file', file: drawer.files[0] || null };
  } else {
    if (drawer.uploadType === 'text') {
      drawer.form = { type: 'text', text: drawer.inputText, file: null };
    } else {
      drawer.form = { type: 'file', text: '', file: drawer.files[0] || null };
    }
  }
  drawer.visible = false;
  ElMessage.success({
    message: '数据已保存',
    duration: 1500
  });
};

const handleSuccess = (type) => (response, file, fileList) => {
  if (type === 'dynamicData' && !file.name.endsWith('.csv')) {
    ElMessage.error({
      message: '文件格式错误，应为csv文件!',
      duration: 1500
    });
    fileList.splice(fileList.indexOf(file), 1);
    return;
  }
  drawers[type].files = fileList.map(f => ({
    filename: f.name,
    filePath: f.response?.filePath || '',
    file_id: f.response?.file_id || ''
  }));
  ElMessage.success({
    message: '文件上传成功!',
    duration: 1500
  });
};

const handleError = (type) => (err, file, fileList) => {
  console.error('Upload error:', err.error);
  ElMessage.error({
    message: '文件上传失败!',
    duration: 1500
  });
};

const handleRemove = (type) => async (file, fileList) => {
  try {
    await axios.post(import.meta.env.VITE_URL + '/delete-file', {
      filePath: file.response?.filePath || '',
      file_id: file.response?.file_id || ''
    });
    ElMessage.success({
      message: '文件删除成功!',
      duration: 1500
    });
    drawers[type].files = fileList.map(f => ({
      filename: f.name,
      filePath: f.response?.filePath || '',
      file_id: f.response?.file_id || ''
    }));
  } catch (error) {
    console.error('删除文件时出错:', error);
    ElMessage.error({
      message: '文件删除失败!',
      duration: 1500
    });
  }
};

const handleSubmit = async () => {
  const codeForm = drawers.code.form;
  if (!codeForm || !codeForm.type ||
    (codeForm.type === 'text' && (!codeForm.text || codeForm.text.trim() === '')) ||
    (codeForm.type === 'file' && !codeForm.file)) {
    ElMessage.error({
      message: '请上传代码!',
      duration: 1500
    });
    return;
  }
  try {
    let userId = localStorage.getItem("userId");
    let data = {
      code: drawers.code.form,
      ir: drawers.ir.form,
      cfg: drawers.cfg.form,
      dynamicData: drawers.dynamicData.form,
      user_id: userId
    };
    ElMessage.success({
      message: '上传成功',
      duration: 1500
    });
    const response = await axios.post(import.meta.env.VITE_URL + '/mt_model', data);
    handleReset();
    result.value = response.data;
    // console.log(result.value);
    ElMessage.success({
      message: '分析成功',
      duration: 1500
    });
  } catch (error) {
    ElMessage.error({
      message: '请求失败，稍后重试',
      duration: 1500
    });
    console.error('请求错误:', error);
  }
};
const handleEvaluate = async () => {
  // const codeForm = drawers.code.form;
  // if (!codeForm || !codeForm.type ||
  //   (codeForm.type === 'text' && (!codeForm.text || codeForm.text.trim() === '')) ||
  //   (codeForm.type === 'file' && !codeForm.file)) {
  //   ElMessage.error({
  //     message: '请上传代码!',
  //     duration: 1500
  //   });
  //   return;
  // }
  try {
    let userId = localStorage.getItem("userId");
    let data = {
      code: drawers.code.form,
      ir: drawers.ir.form,
      cfg: drawers.cfg.form,
      dynamicData: drawers.dynamicData.form,
      user_id: userId
    };
    const response = await axios.post(import.meta.env.VITE_URL + '/mt_evaluate', data);
    let value1 = result.value.model_process
    let value2 = result.value.loss_image
    handleReset()
    result.value.evaluate_image1 = response.data.evaluate_image1
    result.value.evaluate_image2 = response.data.evaluate_image2
    result.value.evaluate_image3 = response.data.evaluate_image3
    result.value.model_process = value1
    result.value.loss_image = value2

    ElMessage.success({
      message: '分析成功',
      duration: 1500
    });
  } catch (error) {
    ElMessage.error({
      message: '请求失败，稍后重试',
      duration: 1500
    });
    console.error('请求错误:', error);
  }
};

const handleReset = () => {
  Object.values(drawers).forEach(drawer => {
    drawer.inputText = '';
    drawer.selectedOption = '';
    drawer.files = [];
    drawer.form = {};
  });
  result.value.model_process = '';
  result.value.loss_image = '';
  result.value.evaluate_image1 = '';
  result.value.evaluate_image2 = '';
  result.value.evaluate_image3 = '';
  if (uploadRefCode.value) {
    uploadRefCode.value.clearFiles();
  }
  if (uploadRefIr.value) {
    uploadRefIr.value.clearFiles();
  }
  if (uploadRefCfg.value) {
    uploadRefCfg.value.clearFiles();
  }
  if (uploadRefDynamicData.value) {
    uploadRefDynamicData.value.clearFiles();
  }
};

onUnmounted(() => {
  Object.values(drawers).forEach(drawer => {
    if (drawer.files.length > 0) {
      handleRemove(drawer.files[0]);
    }
  });
});
</script>

<style scoped>
@import url('/src/assets/css/base.css');
@import url('/src/assets/css/dcu.css');
</style>