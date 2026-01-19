<template>
    <div class="foot-container">
      <!-- 模块1: 调用GPT接口 -->
      <div class="container">
        <h4>模块一：插桩智能体</h4>
        <div class="form-section">
          <label for="textarea" class="title-label">请输入待插桩源代码：</label>
          <el-input type="textarea" v-model="question1" rows="5" class="textarea-input"></el-input>
          <div class="button-section">
            <el-button type="primary" @click="submitQuestion1">提交</el-button>
            <el-button @click="resetQuestion1">重置</el-button>
          </div>
        </div>
        <div class="result-section">
          <h3 class="title-label">结果</h3>
          <p>{{ answer1 }}</p>
        </div>
      </div>
  
      <!-- 模块2: 策略计算 -->
      <div class="container">
        <h4>模块二：策略生成</h4>
        <div class="form-section">
          <div>
            <label for="dataBlockSize" class="tip-label">数据块大小 (bytes):</label>
            <el-input v-model="dataBlockSize" placeholder="请输入数据块大小" class="input-field"></el-input>
          </div>
          <div>
            <label for="accessStride" class="tip-label">访问步长:</label>
            <el-input v-model="accessStride" placeholder="请输入访问步长" class="input-field"></el-input>
          </div>
          <div>
            <label for="accessFrequency" class="tip-label">访问频率 (%):</label>
            <el-input v-model="accessFrequency" placeholder="请输入访问频率" class="input-field"></el-input>
          </div>
          <div class="button-section">
            <el-button type="primary" @click="calculateStrategy">分析策略</el-button>
            <el-button @click="resetStrategy">重置</el-button>
          </div>
        </div>
        <div class="result-section">
          <h3 class="title-label">结果</h3>
          <p>策略: {{ strategy }}</p>
          <p>策略参数: {{ strategyParams }}</p>
        </div>
      </div>
  
      <!-- 模块3: 调用GPT接口 -->
      <div class="container">
        <h4>模块三：代码优化智能体</h4>
        <div class="form-section">
          <label for="textarea" class="title-label">请输入使用的策略以及策略参数：</label>
          <el-input type="textarea" v-model="question3" rows="5" class="textarea-input"></el-input>
          <div class="button-section">
            <el-button type="primary" @click="submitQuestion3">提交</el-button>
            <el-button @click="resetQuestion3">重置</el-button>
          </div>
        </div>
        <div class="result-section">
          <h3 class="title-label">优化代码</h3>
          <p>{{ answer3 }}</p>
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
  import { ref } from 'vue';
  import axios from 'axios';
  
  // 模块一：问答系统
  const question1 = ref('');
  const answer1 = ref('');
  
  const submitQuestion1 = async () => {
    if (!question1.value.trim()) {
      ElMessage.error('请输入问题');
      return;
    }
    try {
      const response = await axios.post('https://yunwu.ai/v1/chat/completions', {
        model: "o1-preview",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: question1.value },
        ],
        temperature: 1,
      }, {
        headers: {
          Authorization: `Bearer sk-KLcFv0I4KYwMpeszKf31IDzEgfctJBKAsTRUC5dgmn7rkDHb`
        }
      });
      answer1.value = response.data.choices[0].message.content;
    } catch (error) {
      console.error(error);
      ElMessage.error('请求失败，请稍后再试');
    }
  };
  
  const resetQuestion1 = () => {
    question1.value = '';
    answer1.value = '';
  };
  
  // 模块二：策略计算
  const dataBlockSize = ref('');
  const accessStride = ref('');
  const accessFrequency = ref('');
  const strategy = ref('');
  const strategyParams = ref('');
  
  const calculateStrategy = () => {
    if (!dataBlockSize.value || !accessStride.value || !accessFrequency.value) {
      ElMessage.error('请填写所有输入项');
      return;
    }
    // 示例策略逻辑（可根据实际需求修改）
    strategy.value = '缓存优先策略';
    strategyParams.value = `数据块大小: ${dataBlockSize.value} bytes, 访问步长: ${accessStride.value}, 访问频率: ${accessFrequency.value}%`;
  };
  
  const resetStrategy = () => {
    dataBlockSize.value = '';
    accessStride.value = '';
    accessFrequency.value = '';
    strategy.value = '';
    strategyParams.value = '';
  };
  
  // 模块三：高级问答系统
  const question3 = ref('');
  const answer3 = ref('');
  
  const submitQuestion3 = async () => {
    if (!question3.value.trim()) {
      ElMessage.error('请输入问题');
      return;
    }
    try {
      const response = await axios.post('https://yunwu.ai/v1/chat/completions', {
        model: "o1-preview",
        messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: question3.value },
        ],
        temperature: 1,
      }, {
        headers: {
          Authorization: `Bearer sk-KLcFv0I4KYwMpeszKf31IDzEgfctJBKAsTRUC5dgmn7rkDHb`
        }
      });
      answer3.value = response.data.choices[0].message.content;
    } catch (error) {
      console.error(error);
      ElMessage.error('请求失败，请稍后再试');
    }
  };
  
  const resetQuestion3 = () => {
    question3.value = '';
    answer3.value = '';
  };
  </script>
  
  <style scoped>
  @import url('/src/assets/css/base.css');
  /* 跟实体抽取页面样式一样 */
  @import url('/src/assets/css/dcu.css');
  
  .container {
    margin-bottom: 20px;
  }
  
  .form-section {
    margin-bottom: 10px;
  }
  
  .result-section {
    border-top: 1px solid #ccc;
    padding-top: 10px;
  }
  
  .input-field {
    width: 100%;
    margin-bottom: 10px;
  }
  </style>