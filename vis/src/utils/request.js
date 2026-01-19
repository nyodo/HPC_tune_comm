import axios from 'axios'
import { ElMessage } from 'element-plus'
import Router from '../router'

let hostURL = import.meta.env.VITE_URL;

// 创建axios实例
const request = axios.create({
  baseURL: hostURL,
  timeout: 15000,// 请求超时时间
})

//响应拦截器
request.interceptors.response.use(res => {
  if (!!res.data.code && res.data.code != 200) {
    if (res.data.code === 401) {
      Router.push("/login") //跳转到登陆页面
      localStorage.clear()
      ElMessage({
        showClose: true,
        message: res.data.data || "未知错误",
        type: 'warning',
      })
    }
  }
  return res
}, error => {
  const status = error.response?.status;
  const errorMessage = error.response?.data?.message || error.response?.data?.error;
  ElMessage({
    showClose: true,
    message: `${status ? `请求失败(${status})` : '请求失败'}${errorMessage ? `: ${errorMessage}` : ''}`,
    type: 'error',
  })
  return Promise.reject(error);
});

//请求拦截器
request.interceptors.request.use(config => {
  let token = localStorage.getItem("token");
  if (token) {
    config.headers['Ac-Token'] = token
  }
  return config
}, error => {
  console.log('请求出错了', error)
});

export default request;