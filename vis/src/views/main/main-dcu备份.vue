<template>
  <div class="foot-container">
    <div class="container">
      <h4>DCU平台建模调优</h4>
      <div class="form-section">
        <div>
          <div style="display: flex;justify-content: space-between;">
            <p class="small-text">上传核函数代码或文件</p>
            <el-button type="primary" @click="handleChange">切换模式</el-button>
          </div>
          <label for="inputText" class="tip-label">
            <!-- (本系统支持预定义的实体类型抽取，包括LOC, ORG, PER) -->&nbsp;
          </label>
        </div>
        <div>
          <label for="textarea" class="title-label">{{ labelText }}</label>
          <el-select v-show="mode === 'text'" v-model="selectedOption" placeholder="可选择样例或输入代码" class="select-input">
            <el-option v-for="option in options" :key="option.value"  :label="option.label" :value="option.cppCode"></el-option>
          </el-select>
          <el-input v-show="mode === 'text'" type="textarea" v-model="inputText" rows="10"
            class="textarea-input"></el-input>
          <el-upload v-show="mode === 'file'" class="upload-demo" drag :action="uploadFileUrl" :limit="1" accept=".txt,.xls,.xlsx"
            :on-success="handleSuccess" :on-error="handleError" :on-remove="handleRemove" ref="uploadRef">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                一次提问限一个txt文件或xls、xlsx文件
              </div>
            </template>
          </el-upload>
        </div>
        <div class="button-section">
          <el-button type="primary" @click="handleSubmit">提交</el-button>
          <el-button @click="handleReset">重置</el-button>
        </div>
      </div>
      <div class="result-section">
        <h3 class="title-label">建模分析结果</h3>
        <div class="result-content">
          <!-- <p v-if="mode === 'text' && textResult.text">文本: {{ textResult.text }}</p> -->
          <p v-if="mode === 'text' && textResult.analysis">{{ textResult.analysis }}</p>
          <p v-if="mode === 'file' && fileResult.analysis">{{ fileResult.analysis }}</p>
        </div>
      </div>
      <div class="result-section">
        <h3 class="title-label">调优结果</h3>
        <div class="result-content">
          <p v-if="mode === 'text' && textResult.improve" v-html="renderMarkdown(textResult.improve)"></p>
          <p v-if="mode === 'file' && fileResult.improve" v-html="renderMarkdown(fileResult.improve)"></p>
        </div>
      </div>
    </div>

    <!-- 页脚 -->
    <div id="footer">
      <div class="copyright"><span> Copyright&copy; 2025 <a href="https://se.xjtu.edu.cn/" target="_blank">XJTU</a> All
          Rights Reserved</span>
      </div>
      <div class="credits"></div>
    </div>
  </div>

</template>

<script setup>
import { ref, watch,onUnmounted } from 'vue';
import { ElMessage } from 'element-plus';
import axios from 'axios';
import MarkdownIt from 'markdown-it'

// 处理Markdown字符串
const md = new MarkdownIt()
const renderMarkdown = (text) => {
    return md.render(text)
}

const uploadFileUrl = ref(import.meta.env.VITE_URL + '/upload-file');
const inputText = ref('');
const selectedOption = ref('');

const options = ref([
  { 
    value: 'option1', 
    label: '样例1：ATAX核函数',
    cppCode: `__global__ void atax_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp){\n\tint i = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (i < _PB_NX){\n\t\ttmp[i] = 0;\n\t\tint j;\n\t\tfor(j=0; j < _PB_NY; j++){\n\t\t\ttmp[i] += A[i*NY+j] * x[j];\n\t\t}\n\t}}`
  },
  {
    value: 'option2',
    label: '样例2：BICG核函数',
    cppCode: `__global__ void bicg_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_NY){\n\t\ts[j] = 0.0f;\n\t\tint i;\n\t\tfor(i = 0; i < _PB_NX; i++){\n\t\t\ts[j] += r[i] * A[i * NY + j];\n\t\t}}}\n\n__global__ void bicg_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q){\n\tint i = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (i < _PB_NX){\n\t\tq[i] = 0.0f;\n\t\tint j;\n\t\tfor(j=0; j < _PB_NY; j++){\n\t\t\tq[i] += A[i * NY + j] * p[j];\n\t\t}}}`
  },
  {
    value: 'option3',
    label: '样例3：CORRELATION核函数',
    cppCode: `__global__ void mean_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_M){\n\t\tmean[j] = 0.0;\n\t\tint i;\n\t\tfor(i=0; i < _PB_N; i++){\n\t\t\tmean[j] += data[i*M + j];\n\t\t}\n\t\tmean[j] /= (DATA_TYPE)FLOAT_N;}}\n\n__global__ void std_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tif (j < _PB_M){\n\t\tstd[j] = 0.0;\n\t\tint i;\n\t\tfor(i = 0; i < _PB_N; i++){\n\t\t\tstd[j] += (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j]);\n\t\t}\n\t\tstd[j] /= (FLOAT_N);\n\t\tstd[j] = sqrt(std[j]);\n\t\tif(std[j] <= EPS){\n\t\t\tstd[j] = 1.0;}}}\n\n__global__ void reduce_kernel(int m, int n, DATA_TYPE *mean, DATA_TYPE *std, DATA_TYPE *data){\n\tint j = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i = blockIdx.y * blockDim.y + threadIdx.y;\n\tif ((i < _PB_N) && (j < _PB_M)){\n\t\tdata[i*M + j] -= mean[j];\n\t\tdata[i*M + j] /= (sqrt(FLOAT_N) * std[j]);}}\n\n__global__ void corr_kernel(int m, int n, DATA_TYPE *symmat, DATA_TYPE *data){\n\tint j1 = blockIdx.x * blockDim.x + threadIdx.x;\n\tint i, j2;\n\tif (j1 < (_PB_M-1)){\n\t\tsymmat[j1*M + j1] = 1.0;\n\t\tfor (j2 = (j1 + 1); j2 < _PB_M; j2++){\n\t\t\tsymmat[j1*M + j2] = 0.0;\n\t\t\tfor(i = 0; i < _PB_N; i++){\n\t\t\t\tsymmat[j1*M + j2] += data[i*M + j1] * data[i*M + j2];\n\t\t\t}\n\t\t\tsymmat[j2*M + j1] = symmat[j1*M + j2];}}}`
  }
]);
watch(selectedOption, (newValue) => {
  if (newValue) {
    // console.log(newValue)
    inputText.value = newValue;
  }
});

// 上传文件
const files = ref([]);
const uploadRef = ref(null);

const handleSuccess = (response, file, fileList) => {
  // 更新 files
  files.value = fileList.map(f => ({
    filename: f.name,
    filePath: f.response?.filePath || '',
    file_id: f.response?.file_id || ''
  }));
  ElMessage.success({
    message: '文件上传成功!',
    duration: 1500
  });
};
const handleError = (err, file, fileList) => {
  // 处理上传失败的情况
  console.error('Upload error:', err.error);
  // 可以在这里更新 UI、通知用户等
  ElMessage.error({
    message: '文件上传失败!',
    duration: 1500
  });
};
const handleRemove = async (file, fileList) => {
  try {
    // 发送删除请求到后端
    await axios.post(import.meta.env.VITE_URL + '/delete-file', {
      filePath: file.response?.filePath || '', // 假设文件路径在 file.response.filePath 中
      file_id: file.response?.file_id || ''
    });
    ElMessage.success({
      message: '文件删除成功!',
      duration: 1500
    });
    // 更新 files
    files.value = fileList.map(f => ({
    filename: f.name,
    filePath: f.response?.filePath || '',
    file_id: f.response?.file_id || ''
  }));
    console.log(files.value)
  } catch (error) {
    console.error('删除文件时出错:', error);
    ElMessage.error({
      message: '文件删除失败!',
      duration: 1500
    });
  }
};

// 页面呈现的返回结果
const textResult = ref({
  text: '',
  analysis: '',
  improve: '',
});
const fileResult = ref({
  analysis: '',
  improve: '',
});
const labelText = ref("代码");
const mode = ref("text");

const handleSubmit = async () => {
  try {
    let userId = localStorage.getItem("userId");
    if (mode.value == 'text') {
      if (!inputText.value) {
        ElMessage.warning({
          message: '输入为空',
          duration: 1500
        });
        return;
      }
      const response = await axios.post(import.meta.env.VITE_URL + '/dcu_code', { text: inputText.value, user_id: userId });
      textResult.value = response.data;
      console.log(textResult.value)
    } else {
      if(files.value.length == 0){
        ElMessage.warning({
          message: '输入为空',
          duration: 1500
        });
        return;
      }
      const response = await axios.post(import.meta.env.VITE_URL + '/file_dcu_code', { file_path: files.value[0].filePath,file_id: files.value[0].file_id, user_id: userId });
      fileResult.value = response.data;
      console.log(fileResult.value)
    }

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

function handleChange() {
  mode.value = mode.value === 'text' ? 'file' : 'text';
  labelText.value = mode.value === 'text' ? '文本' : '文件';
}
async function handleReset() {
  if (mode.value == 'text') {
    // 清空输入和选择内容
    inputText.value = '';
    selectedOption.value = '';
    textResult.value.text = '';
    textResult.value.analysis = '';
    textResult.value.improve = '';
  } else {
    checkFile();
  }
}
async function checkFile() {
    if (fileResult.value.analysis == '' && files.value.length != 0) {
      try {
        // 发送删除请求到后端
        await axios.post(import.meta.env.VITE_URL + '/delete-file', {
          filePath: files.value.length == 0 ? "" : files.value[0].filePath,
          file_id: files.value.length == 0 ? "" : files.value[0].file_id
        });
      } catch (error) {
        console.error('删除文件时出错:', error);
      }
    }
    files.value = []; // 清空上传文件列表
    if (uploadRef.value) {
      uploadRef.value.clearFiles(); // 清空上传组件的文件列表
    }
    fileResult.value.analysis = '';
    fileResult.value.improve = '';
}
onUnmounted(() => {
  checkFile();
});
</script>

<style scoped>
@import url('/src/assets/css/base.css');
@import url('/src/assets/css/dcu.css');
</style>
