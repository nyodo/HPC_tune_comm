<template>

  <!-- ======= About Us Section ======= -->
  <section id="about" class="about" data-aos="fade-up">
    <div class="container">

      <div class="row no-gutters">
        <div class="col-lg-6 video-box">
          <img src="/src/assets/img/about.png" class="img-fluid" alt="">
          <a href="/src/assets/video.mp4" class="venobox play-btn mb-4" data-vbtype="iframe" data-autoplay="true"></a>
        </div>

        <div class="col-lg-6 d-flex flex-column justify-content-center about-content">

          <div class="section-title">
            <h2>系统演示</h2>
            <p>系统基于Flask框架，前端页面由HTML+CSS+JS编写，后端由Python支持。使用Python爬虫实时获取最新的电影数据，交由大语言模型分析。使用Echarts和WordCloud等技术进行可视化
            </p>
          </div>

          <div class="icon-box">
            <div class="icon"><i class="bx bx-fingerprint"></i></div>
            <h4 class="title"><a href="">Python爬虫</a></h4>
            <p class="description">Python语言在爬虫领域非常流行，因为它具有简洁易读的语法、强大的库支持和广泛的应用。Requests、Beautiful
              Soup和Scrapy等库，可用于发送HTTP请求，获取网页内容，解析网页内容，获得所需数据，使爬虫编程变得更简单易懂。</p>
          </div>

          <div class="icon-box">
            <div class="icon"><i class="bx bx-gift"></i></div>
            <h4 class="title"><a href="">大模型</a></h4>
            <p class="description">
              目前大模型在搜索引擎、写作创作、编写程序、嵌入式设备等特定领域都有垂直应用，极大提升了用户的生活体验及工作效率。而大模型在电影数据分析方面的应用较少，有广阔的市场前景和较好的发展潜力。</p>
          </div>

        </div>
      </div>

    </div>
  </section><!-- End About Us Section -->

  <!-- 页脚 -->
  <footer id="footer">
    <div class="container">
      <div class="copyright"><span> Copyright&copy; 2025 <a href="https://se.xjtu.edu.cn/" target="_blank">XJTU</a> All Rights Reserved</span>
      </div>
      <div class="credits"></div>
    </div>
  </footer>
</template>

<script setup>
import { onMounted} from 'vue';

onMounted(() => {

});
</script>


<style scoped>
@import url('/src/assets/vendor/boxicons/css/boxicons.min.css');
@import url('/src/assets/css/base.css');

/*--------------------------------------------------------------
  # About Us
  --------------------------------------------------------------*/
  .about {
    padding-bottom: 30px;
  }
  
  .about .container {
    box-shadow: 0 5px 25px 0 rgba(214, 215, 216, 0.6);
  }
  
  .about .video-box img {
    padding: 15px 0;
  }
  
  .about .section-title p {
    text-align: left;
    font-style: italic;
    color: #666;
  }
  
  .about .about-content {
    padding: 40px;
  }
  
  .about .icon-box + .icon-box {
    margin-top: 40px;
  }
  
  .about .icon-box .icon {
    float: left;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 72px;
    height: 72px;
    background: #f1f7fb;
    border-radius: 6px;
    transition: 0.5s;
  }
  
  .about .icon-box .icon i {
    color: #428bca;
    font-size: 32px;
  }
  
  .about .icon-box:hover .icon {
    background: #428bca;
  }
  
  .about .icon-box:hover .icon i {
    color: #fff;
  }
  
  .about .icon-box .title {
    margin-left: 95px;
    font-weight: 700;
    margin-bottom: 10px;
    font-size: 18px;
    text-transform: uppercase;
  }
  
  .about .icon-box .title a {
    color: #343a40;
    transition: 0.3s;
  }
  
  .about .icon-box .title a:hover {
    color: #428bca;
  }
  
  .about .icon-box .description {
    margin-left: 95px;
    line-height: 24px;
    font-size: 14px;
  }
  
  .about .video-box {
    position: relative;
  }
  
  .about .play-btn {
    width: 94px;
    height: 94px;
    background: radial-gradient(#428bca 50%, rgba(66, 139, 202, 0.4) 52%);
    border-radius: 50%;
    display: block;
    position: absolute;
    left: calc(50% - 47px);
    top: calc(50% - 47px);
    overflow: hidden;
  }
  
  .about .play-btn::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translateX(-40%) translateY(-50%);
    width: 0;
    height: 0;
    border-top: 10px solid transparent;
    border-bottom: 10px solid transparent;
    border-left: 15px solid #fff;
    z-index: 100;
    transition: all 400ms cubic-bezier(0.55, 0.055, 0.675, 0.19);
  }
  
  .about .play-btn::before {
    content: '';
    position: absolute;
    width: 120px;
    height: 120px;
    -webkit-animation-delay: 0s;
    animation-delay: 0s;
    -webkit-animation: pulsate-btn 2s;
    animation: pulsate-btn 2s;
    -webkit-animation-direction: forwards;
    animation-direction: forwards;
    -webkit-animation-iteration-count: infinite;
    animation-iteration-count: infinite;
    -webkit-animation-timing-function: steps;
    animation-timing-function: steps;
    opacity: 1;
    border-radius: 50%;
    border: 5px solid rgba(66, 139, 202, 0.7);
    top: -15%;
    left: -15%;
    background: rgba(198, 16, 0, 0);
  }
  
  .about .play-btn:hover::after {
    border-left: 15px solid #428bca;
    transform: scale(20);
  }
  
  .about .play-btn:hover::before {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translateX(-40%) translateY(-50%);
    width: 0;
    height: 0;
    border: none;
    border-top: 10px solid transparent;
    border-bottom: 10px solid transparent;
    border-left: 15px solid #fff;
    z-index: 200;
    -webkit-animation: none;
    animation: none;
    border-radius: 0;
  }
  

</style>