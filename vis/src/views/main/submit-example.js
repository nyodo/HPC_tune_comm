// 这是一个将表单数据提交到后端的示例代码
// 在实际项目中，你可以将这些代码集成到你的startCollection方法中

import axios from 'axios';
import { ElMessage } from 'element-plus';

// 假设你有一个ioForm对象，包含了所有的表单数据
const ioForm = ref({
  // ... 表单字段
});

// 表单提交方法
const submitFormToBackend = async () => {
  try {
    // 1. 验证表单数据
    if (!ioForm.value.name) {
      ElMessage.error('请输入作业名称');
      return;
    }
    
    if (!ioForm.value.result_folder) {
      ElMessage.error('请输入结果文件夹路径');
      return;
    }
    
    // 2. 准备请求数据
    const requestData = {
      ...ioForm.value
      // 可以在这里对数据进行转换或处理
    };
    
    // 3. 发送请求到后端
    const response = await axios.post('/api/io-collection', requestData, {
      headers: {
        'Content-Type': 'application/json'
      },
      // 如果需要身份验证，可以在这里添加token
      // Authorization: `Bearer ${token}`
    });
    
    // 4. 处理成功响应
    ElMessage.success('表单提交成功');
    console.log('后端返回数据:', response.data);
    
    // 可以在这里执行其他操作，比如重置表单、跳转到其他页面等
    // resetIOForm();
    
  } catch (error) {
    // 5. 处理错误
    ElMessage.error('表单提交失败');
    console.error('提交错误:', error);
    
    // 可以根据错误类型进行更详细的处理
    if (error.response) {
      // 服务器返回了错误状态码
      console.error('服务器错误:', error.response.data);
      console.error('状态码:', error.response.status);
    } else if (error.request) {
      // 请求已发送但没有收到响应
      console.error('网络错误:', error.request);
    } else {
      // 请求配置时发生错误
      console.error('请求配置错误:', error.message);
    }
  }
};

// 在startCollection方法中调用提交方法
const startCollection = async (type) => {
  if (type === 'io') {
    await submitFormToBackend();
  } else {
    ElMessage.success(`开始${type}数据采集`);
  }
};
