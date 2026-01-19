<template>
<div class="foot-container">
  <!-- ======= 技术介绍部分 ======= -->
  <section id="faq" class="faq section-bg" data-aos="fade-up">
    <div class="container">

      <div class="section-title">
        <h2>使用说明</h2>
      </div>

      <div class="row  d-flex align-items-stretch">
        <span v-html="renderMarkdown(intro)"></span>
      </div>

    </div>
  </section><!-- End 技术介绍部分 -->

  <!-- 页脚 -->
  <footer id="footer">
    <div class="container">
      <div class="copyright"><span> Copyright&copy; 2025 <a href="https://se.xjtu.edu.cn/" target="_blank">XJTU</a> All Rights Reserved</span>
      </div>
      <div class="credits"></div>
    </div>
  </footer>
</div>
</template>

<script setup>
import { onMounted,ref} from 'vue';
import MarkdownIt from 'markdown-it'

// 处理Markdown字符串
const md = new MarkdownIt()
const renderMarkdown = (text) => {
    return md.render(text)
}
const intro = ref(
  `#### DCU平台建模调优

此模块有两种模式：
1. 单个文本提交，将需要优化的代码粘贴进输入框。
2. 单个文件提交，将需要优化的代码文件提交。

#### 用户管理

此模块仅对管理员开放，功能如下：
1. 增删改查用户信息。
2. 增加管理员。

#### DCU分析记录

此模块仅对管理员开放，功能如下：
1. 增删改查用户DCU平台代码提交记录。
2. 后续考虑对普通用户开放，需要调整代码逻辑。

#### 文件记录

此模块仅对管理员开放，查询用户上传文件的相关信息。

`)
onMounted(() => {

});
</script>


<style scoped>
@import url('/src/assets/vendor/boxicons/css/boxicons.min.css');
@import url('/src/assets/css/base.css');

/*--------------------------------------------------------------
  # Frequently Asked Questions
  --------------------------------------------------------------*/
  .faq {
    padding-bottom: 30px;
  }
  
  .faq .faq-item {
    margin-bottom: 40px;
  }
  
  .faq .faq-item h4 {
    font-size: 20px;
    line-height: 28px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #1f3548;
  }
</style>