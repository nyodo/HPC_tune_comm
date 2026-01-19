<template>
<div class="foot-container">
  <!-- ======= 联系作者部分 ======= -->
  <section id="contact" class="contact" data-aos="fade-up">
    <div class="container">
      <div class="section-title">
        <h2>联系我们</h2>
      </div>
      <div class="row">
        <div class="col-lg-6 d-flex align-items-stretch">
          <div class="info-box">
            <i class="bx bx-map"></i>
            <h3>我的地址</h3>
            <p>xxx</p>
          </div>
        </div>
        <div class="col-lg-3 d-flex align-items-stretch">
          <div class="info-box">
            <i class="bx bx-envelope"></i>
            <h3>我的邮箱</h3>
            <p>xxx<br></p>
          </div>
        </div>
        <div class="col-lg-3 d-flex align-items-stretch">
          <div class="info-box">
            <i class="bx bx-phone-call"></i>
            <h3>我的电话</h3>
            <p>xxx<br></p>
          </div>
        </div>
        <div class="col-lg-12">
          <form @submit.prevent="submitForm" role="form" class="php-email-form" id="contactForm">
            <div class="form-row">
              <div class="col-lg-6 form-group">
                <input type="text" v-model="form.name" class="form-control" id="name" placeholder="您的姓名" />
                <div class="validate"></div>
              </div>
              <div class="col-lg-6 form-group">
                <input type="email" v-model="form.email" class="form-control" id="email" placeholder="您的邮箱" />
                <div class="validate"></div>
              </div>
            </div>
            <div class="form-group">
              <input type="text" v-model="form.subject" class="form-control" id="subject" placeholder="主题" />
              <div class="validate"></div>
            </div>
            <div class="form-group">
              <textarea v-model="form.message" class="form-control" rows="5" placeholder="内容"></textarea>
              <div class="validate"></div>
            </div>
            <div class="mb-3">
              <div class="loading" v-if="loading">Loading</div>
              <div class="error-message" v-if="errorMessage">{{ errorMessage }}</div>
              <div class="sent-message" v-if="sentMessage">Your message has been sent. Thank you!</div>
            </div>
            <div class="text-center"><el-button color="#003f88" :dark="isDark" type="primary" @click="submitForm">提&nbsp;&nbsp;&nbsp;交</el-button></div>
          </form>
        </div>
      </div>
    </div>
  </section><!-- End 联系作者部分 -->


  <!-- 页脚 -->
  <div id="footer">
      <div class="copyright"><span> Copyright&copy; 2025 <a href="https://se.xjtu.edu.cn/" target="_blank">XJTU</a> All Rights Reserved</span>
      </div>
      <div class="credits"></div>
  </div>
</div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { ElMessage } from 'element-plus';

const form = ref({
  name: '',
  email: '',
  subject: '',
  message: ''
});

const loading = ref(false);
const errorMessage = ref('');
const sentMessage = ref(false);

const submitForm = async () => {
  if (!form.value.message) {
    ElMessage.warning({
    message: '请填写内容！',
    duration: 1500  
  });
    return;
  }

  loading.value = true;
  errorMessage.value = '';
  sentMessage.value = false;

  try {
    const response = await fetch(import.meta.env.VITE_URL+'/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(form.value)
    });

    if (response.ok) {
      const data = await response.json();
      console.log(data);
      form.value = {
        name: '',
        email: '',
        subject: '',
        message: ''
      };
      sentMessage.value = true;
      ElMessage.success({
        message: '提交成功！',
        duration: 1500
      });
    } else {
      throw new Error('Form submission failed');
    }
  } catch (error) {
    console.error(error);
    errorMessage.value = '提交失败，稍后请重试！';
    ElMessage.error({
    message: '提交失败，请稍后重试！',
    duration: 1500  
  });
  } finally {
    loading.value = false;
  }
};
onMounted(() => {

});
</script>


<style scoped>
@import url('/src/assets/vendor/boxicons/css/boxicons.min.css');
@import url('/src/assets/css/base.css');

/*--------------------------------------------------------------
  # Contact Us
  --------------------------------------------------------------*/
  .contact .info-box {
    color: #444;
    text-align: center;
    box-shadow: 0 0 30px rgba(214, 215, 216, 0.6);
    padding: 20px 0 30px 0;
    margin-bottom: 30px;
    width: 100%;
  }
  
  .contact .info-box i {
    font-size: 32px;
    color: #428bca;
    border-radius: 50%;
    padding: 8px;
    border: 2px dotted #9eccf4;
  }
  
  .contact .info-box h3 {
    font-size: 20px;
    color: #666;
    font-weight: 700;
    margin: 10px 0;
  }
  
  .contact .info-box p {
    padding: 0;
    line-height: 24px;
    font-size: 14px;
    margin-bottom: 0;
  }
  
  .contact .php-email-form {
    box-shadow: 0 0 30px rgba(214, 215, 216, 0.6);
    padding: 30px;
  }
  
  .contact .php-email-form .validate {
    display: none;
    color: red;
    margin: 0 0 15px 0;
    font-weight: 400;
    font-size: 13px;
  }
  
  .contact .php-email-form .error-message {
    display: none;
    color: #fff;
    background: #ed3c0d;
    text-align: center;
    padding: 15px;
    font-weight: 600;
  }
  
  .contact .php-email-form .sent-message {
    display: none;
    color: #fff;
    background: #18d26e;
    text-align: center;
    padding: 15px;
    font-weight: 600;
  }
  
  .contact .php-email-form .loading {
    display: none;
    background: #fff;
    text-align: center;
    padding: 15px;
  }
  
  .contact .php-email-form .loading:before {
    content: "";
    display: inline-block;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    margin: 0 10px -6px 0;
    border: 3px solid #18d26e;
    border-top-color: #eee;
    -webkit-animation: animate-loading 1s linear infinite;
    animation: animate-loading 1s linear infinite;
  }
  
  .contact .php-email-form input, .contact .php-email-form textarea {
    border-radius: 0;
    box-shadow: none;
    font-size: 14px;
  }
  

  
  .contact .php-email-form input {
    padding: 20px 15px;
  }
  
  .contact .php-email-form textarea {
    padding: 12px 15px;
  }
</style>