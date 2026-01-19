<template>
  <div class="foot-container">
    <div class="container">
      <h4>关系抽取</h4>
      <div class="form-section">
        <div>
          <div style="display: flex;justify-content: space-between;">
            <p class="small-text">输入文本或文件，使用e1和e2标签分别包裹实体1和2</p>
            <el-button type="primary" @click="handleChange">切换模式</el-button>
          </div>
          <label for="inputText" class="tip-label">
            <!-- (本系统支持预定义的关系类型抽取，例如所属专辑, 毕业院校, 导演等) -->&nbsp;
          </label>
        </div>
        <div>
          <label for="textarea" class="title-label">{{ labelText }}</label>
          <el-select v-show="mode === 'text'" v-model="selectedOption" placeholder="可选择样例" class="select-input">
            <el-option v-for="option in options" :key="option.value" :value="option.label"></el-option>
          </el-select>
          <el-input v-show="mode === 'text'" type="textarea" v-model="inputText" rows="5"
            class="textarea-input"></el-input>
          <el-upload v-show="mode === 'file'" class="upload-demo" drag :action="uploadFileUrl"
            :limit="1" accept=".txt" :on-success="handleSuccess" :on-error="handleError" :on-remove="handleRemove"
            ref="uploadRef">
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              拖拽或 <em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                一次提问限一个txt文件
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
        <h3 class="title-label">结果</h3>
        <div class="result-content">
          <p v-if="mode === 'text' && textResult.text">文本: {{ textResult.text }}</p>
          <p v-if="mode === 'text' && textResult.relation">关系: {{ textResult.relation }}</p>
          <p v-if="mode === 'file' && fileResult.text">{{ fileResult.text }}</p>
          <p v-if="mode === 'file' && fileResult.relation">{{ fileResult.relation }}</p>
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


const uploadFileUrl = ref(import.meta.env.VITE_URL+'/upload-file');
const inputText = ref('');
const selectedOption = ref('');

const options = ref([
  { value: 'option1', label: '样例1：2023年11月，由<e1>杭州市人民政府</e1>主办的‘<e2>杭州国际电子商务博览会</e2>’在杭州国际会展中心成功举办，吸引了来自全球的电商企业。' },
  { value: 'option2', label: '样例2：<e2>申请公租房事项</e2>被纳入<e1>国务院办公厅</e1>“高效办成一件事”最新事项清单' },
  { value: 'option3', label: '样例3：‘<e2>中国农村振兴示范项目</e2>’首席负责人<e1>陈宇</e1>，肩负起了推动项目在四川省内的全面落地实施的重任。' },
]);

watch(selectedOption, (newValue) => {
  if (newValue) {
    const startIndex = 4;
    inputText.value = newValue.substring(startIndex);
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
    await axios.post(import.meta.env.VITE_URL+'/delete-file', {
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


const textResult = ref({
  text: '',
  relation: '',
});
const fileResult = ref({
  text: '',
  relation: '',
});
const labelText = ref("文本");
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
      const response = await axios.post(import.meta.env.VITE_URL+'/relation', { text: inputText.value, user_id: userId });
      textResult.value = response.data;
    } else {
      if(files.value.length == 0){
        ElMessage.warning({
          message: '输入为空',
          duration: 1500
        });
        return;
      }
      const response = await axios.post(import.meta.env.VITE_URL+'/batch_relation', { file_path: files.value[0].filePath,file_id: files.value[0].file_id,  user_id: userId });
      fileResult.value = response.data;
    }

    ElMessage.success({
      message: '关系抽取成功',
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
    textResult.value.relation = ''; 

  } else {
    console.log(files)
    checkFile()
  }


}
async function checkFile() {
  if (fileResult.value.text == '' && files.value.length != 0) {
      try {
        // 发送删除请求到后端
        await axios.post(import.meta.env.VITE_URL+'/delete-file', {
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
    fileResult.value.text = '';
    fileResult.value.relation = '';
}
onUnmounted(() => {
  checkFile();
});
</script>

<style scoped>
@import url('/src/assets/css/base.css');
/* 跟实体抽取页面样式一样 */
@import url('/src/assets/css/dcu.css');
</style>