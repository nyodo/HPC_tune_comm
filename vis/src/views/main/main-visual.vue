<template>

  <!-- 宏观数据模块 -->
  <section id="team" class="team">
    <div class="container">
      <div class="section-title">
        <h2>生产经营指标完成情况</h2>
        <p></p>
      </div>
      <!-- ======= Counts Section ======= -->
      <section class="counts section-bg">
        <div class="container">

          <div class="row">

            <div class="col-lg-3 col-md-6 text-center" data-aos="fade-up">
              <div class="count-box">
                <i class="icofont-chart-bar-graph" style="color: #20b38e;"></i>
                <span data-toggle="counter-up">232.61</span>
                <p>产量（万吨）</p>
                <p>完成率：122.43%</p>
              </div>
            </div>

            <div class="col-lg-3 col-md-6 text-center" data-aos="fade-up" data-aos-delay="200">
              <div class="count-box">
                <i class="icofont-pie-chart" style="color: #c042ff;"></i>
                <span data-toggle="counter-up">233.61</span>
                <p>销量（万吨）</p>
                <p>完成率：122.95%</p>
              </div>
            </div>

            <div class="col-lg-3 col-md-6 text-center" data-aos="fade-up" data-aos-delay="400">
              <div class="count-box">
                <i class="icofont-chart-arrows-axis" style="color: #46d1ff;"></i>
                <span data-toggle="counter-up">1860</span>
                <p>生产进尺（米）</p>
                <p>完成率：158.97%</p>
              </div>
            </div>

            <div class="col-lg-3 col-md-6 text-center" data-aos="fade-up" data-aos-delay="600">
              <div class="count-box">
                <i class="icofont-users-alt-5" style="color: #ffb459;"></i>
                <span>综采一队</span>
                <p>三项费用考核</p>
                <p>表现最好单位</p>
              </div>
            </div>

          </div>

        </div>
      </section><!-- End Counts Section -->
    </div>
  </section>

  <!-- 实时每日数据模块 -->
  <section id="realtime" class="realtime section-bg">
    <div class="container clearfix">
      <div class="section-title">
        <h2>每月数据</h2>
      </div>
      <!-- 左边表格 -->
      <div class="left-content leftfix">
        <div class="title clearfix">
          <span class="leftfix">施工数据</span>
          <ul class="filter-list rightfix clearfix">
            <!-- 这里名称修改，js文件里也要改 -->
            <li><a>材料费</a></li>
            <li><a>电费</a></li>
            <li><a>辅助运输费</a></li>
            <li><a>三项费用</a></li>
          </ul>
        </div>
        <!-- 表格 -->
        <table class="cost-table" id="table1">
          <thead>
            <tr>
              <th>单位</th>
              <th>实际产量</th>
              <th>计划</th>
              <th>支出</th>
              <th>节+超-</th>
              <th>节超率</th>
              <th>奖+罚-</th>
            </tr>
          </thead>
          <tbody>
            <!-- 合计行 -->
            <tr>
              <td>合计</td>
              <td></td>
              <td>596.39</td>
              <td>572.88</td>
              <td>23.50</td>
              <td>3.94%</td>
              <td>6.67</td>
            </tr>
            <!-- 综采一队 -->
            <tr>
              <td>综采一队</td>
              <td>1054100</td>
              <td>129.70</td>
              <td>122.52</td>
              <td>7.17</td>
              <td>5.53%</td>
              <td>2.15</td>
            </tr>
            <!-- 综采二队 -->
            <tr>
              <td>综采二队</td>
              <td>1221140</td>
              <td>150.25</td>
              <td>146.16</td>
              <td>4.08</td>
              <td>2.72%</td>
              <td>0.00</td>
            </tr>
            <!-- 掘进一队 -->
            <tr>
              <td>掘进一队</td>
              <td>392</td>
              <td>75.75</td>
              <td>79.29</td>
              <td>-3.54</td>
              <td>-4.67%</td>
              <td>0.00</td>
            </tr>
            <!-- 掘进二队 -->
            <tr>
              <td>掘进二队</td>
              <td>323</td>
              <td>62.42</td>
              <td>56.88</td>
              <td>5.54</td>
              <td>8.87%</td>
              <td>1.66</td>
            </tr>
            <!-- 探放水队 -->
            <tr>
              <td>探放水队</td>
              <td>2326056</td>
              <td>11.07</td>
              <td>10.26</td>
              <td>0.81</td>
              <td>7.33%</td>
              <td>0.24</td>
            </tr>
            <!-- 机电队 -->
            <tr>
              <td>机电队</td>
              <td>2326056</td>
              <td>52.03</td>
              <td>51.30</td>
              <td>0.74</td>
              <td>1.41%</td>
              <td>0.00</td>
            </tr>
            <!-- 胶带运输队 -->
            <tr>
              <td>胶带运输队</td>
              <td>2326056</td>
              <td>90.90</td>
              <td>84.41</td>
              <td>6.49</td>
              <td>7.14%</td>
              <td>1.95</td>
            </tr>
            <!-- 注意：原模板中多了一个“通风维护区”的重复项，这里未做修改，请根据实际情况调整 -->
          </tbody>
        </table>

        <table class="cost-table" id="table2">
          <thead>
            <tr>
              <th>单位</th>
              <th>实际产量</th>
              <th>计划</th>
              <th>支出</th>
              <th>节+超-</th>
              <th>节超率</th>
              <th>奖+罚-</th>
            </tr>
          </thead>
          <tbody>
            <!-- 合计行 -->
            <tr>
              <td>合计</td>
              <td>2035.12</td>
              <td>1971.11</td>
              <td>64.01</td>
              <td>3.15</td>
              <td>3.15%</td>
              <td>29.03</td>
            </tr>
            <!-- 综采一队 -->
            <tr>
              <td>综采一队</td>
              <td>1054100</td>
              <td>273.64</td>
              <td>256.45</td>
              <td>17.19</td>
              <td>6.28%</td>
              <td>7.79</td>
            </tr>
            <!-- 综采二队 -->
            <tr>
              <td>综采二队</td>
              <td>1221140</td>
              <td>315.79</td>
              <td>303.04</td>
              <td>12.75</td>
              <td>4.04%</td>
              <td>2.49</td>
            </tr>
            <!-- 掘进一队 -->
            <tr>
              <td>掘进一队</td>
              <td>392</td>
              <td>133.82</td>
              <td>142.41</td>
              <td>-8.59</td>
              <td>-6.42%</td>
              <td>-3.92</td>
            </tr>
            <!-- 掘进二队 -->
            <tr>
              <td>掘进二队</td>
              <td>323</td>
              <td>110.22</td>
              <td>109.59</td>
              <td>0.63</td>
              <td>0.57%</td>
              <td>-1.48</td>
            </tr>
            <!-- 探放水队 -->
            <tr>
              <td>探放水队</td>
              <td>2326056</td>
              <td>182.46</td>
              <td>185.63</td>
              <td>-3.16</td>
              <td>-1.73%</td>
              <td>0.24</td>
            </tr>
            <!-- 机电队 -->
            <tr>
              <td>机电队</td>
              <td>2326056</td>
              <td>475.13</td>
              <td>458.64</td>
              <td>16.50</td>
              <td>3.47%</td>
              <td>7.52</td>
            </tr>
            <!-- 胶带运输队 -->
            <tr>
              <td>胶带运输队</td>
              <td>2326056</td>
              <td>429.07</td>
              <td>409.50</td>
              <td>19.57</td>
              <td>4.56%</td>
              <td>8.81 </td>
            </tr>
            <!-- 通风维护区 -->
            <tr>
              <td>胶带运输队</td>
              <td>2326056</td>
              <td>114.98</td>
              <td>105.85</td>
              <td>9.13</td>
              <td>7.94% </td>
              <td>7.58</td>
            </tr>
          </tbody>
        </table>

        <table class="cost-table" id="table3">
          <thead>
            <tr>
              <th>单位</th>
              <th>实际产量</th>
              <th>计划</th>
              <th>支出</th>
              <th>节+超-</th>
              <th>节超率</th>
              <th>奖+罚-</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>film[3]</td>
              <td>film[4]</td>
              <td>film[5]</td>
              <td>film[2]</td>
              <td>film[1]</td>
              <td>film[6]</td>
              <td>film[6]</td>
            </tr>
          </tbody>
        </table>

        <table class="cost-table" id="table4">
          <thead>
            <tr>
              <th>单位</th>
              <th>实际产量</th>
              <th>计划</th>
              <th>支出</th>
              <th>节+超-</th>
              <th>节超率</th>
              <th>奖+罚-</th>
            </tr>
          </thead>
          <tbody>
            <!-- 合计行 -->
            <tr>
              <td>合计</td>
              <td></td>
              <td>2035.12</td>
              <td>1971.11</td>
              <td>64.01</td>
              <td>3.15%</td>
              <td>29.03</td>
            </tr>
            <!-- 综采一队 -->
            <tr>
              <td>综采一队</td>
              <td>1054100</td>
              <td>273.64</td>
              <td>256.45</td>
              <td>17.19</td>
              <td>6.28%</td>
              <td>7.79</td>
            </tr>
            <!-- 综采二队 -->
            <tr>
              <td>综采二队</td>
              <td>1221140</td>
              <td>315.79</td>
              <td>303.04</td>
              <td>12.75</td>
              <td>4.04%</td>
              <td>2.49</td>
            </tr>
            <!-- 掘进一队 -->
            <tr>
              <td>掘进一队</td>
              <td>392</td>
              <td>133.82</td>
              <td>142.41</td>
              <td>-8.59</td>
              <td>-6.42%</td>
              <td>-3.92</td>
            </tr>
            <!-- 掘进二队 -->
            <tr>
              <td>掘进二队</td>
              <td>323</td>
              <td>110.22</td>
              <td>109.59</td>
              <td>0.63</td>
              <td>0.57%</td>
              <td>-1.48</td>
            </tr>
            <!-- 探放水队 -->
            <tr>
              <td>探放水队</td>
              <td>2326056</td>
              <td>182.46</td>
              <td>185.63</td>
              <td>-3.16</td>
              <td>-1.73%</td>
              <td>0.24</td>
            </tr>
            <!-- 机电队 -->
            <tr>
              <td>机电队</td>
              <td>2326056</td>
              <td>475.13</td>
              <td>458.64</td>
              <td>16.50</td>
              <td>3.47%</td>
              <td>7.52</td>
            </tr>
            <!-- 胶带运输队 -->
            <tr>
              <td>胶带运输队</td>
              <td>2326056</td>
              <td>429.07</td>
              <td>409.50</td>
              <td>19.57</td>
              <td>4.56%</td>
              <td>8.81 </td>
            </tr>
            <!-- 通风维护区 -->
            <tr>
              <td>通风维护区</td>
              <td>2326056</td>
              <td>114.98</td>
              <td>105.85</td>
              <td>9.13</td>
              <td>7.94% </td>
              <td>7.58</td>
            </tr>
          </tbody>
        </table>

        <!-- 底部提示 -->
        <div class="foot-tip">
          <span>*数据日期 2025.05.22 (每日14:30更新数据)</span>
        </div>
      </div>
      <!-- 右边两块内容 -->
      <div class="right-content leftfix">
        <div class="title clearfix">
          <span class="leftfix">生产经营指标</span>
          <span class="date rightfix"> 2025.05 </span>
        </div>
        <ul class="top-content">
          <li class="clearfix">
            <img src="/src/assets/img/电影票.png" alt="">
            <span>产量（万吨）</span>
            <span>232.61</span>
          </li>
          <li class="clearfix">
            <img src="/src/assets/img/座位.png" alt="">
            <span>销量（万吨）</span>
            <span>233.61</span>
          </li>
          <li class="clearfix">
            <img src="/src/assets/img/人数.png" alt="">
            <span>生产进尺（米）</span>
            <span>1860</span>
          </li>
        </ul>
        <div class="title clearfix">
          <span class="leftfix">月度三项支出</span>
        </div>
        <div class="bottom-content" id="daily-boxoffice-chart"></div>
        <div class="foot-tip">
          <span>*每日14:30更新数据</span>
        </div>
      </div>
    </div>
  </section>


  <!-- 图片模块 -->
  <section id="portfolio" class="portfolio">
    <div class="container">

      <div class="section-title">
        <h2>分析结果可视化</h2>
        <p>通过数学建模以及大数据分析等技术，对集团的生产经营数据进行分析，呈现以下可视化结果。</p>
      </div>

      <div class="row">
        <div class="col-lg-12">
          <ul id="portfolio-flters">
            <li data-filter="*" class="filter-active">All</li>
            <li data-filter=".filter-yearly">Yearly</li>
            <li data-filter=".filter-monthly">Monthly</li>
            <li data-filter=".filter-daily">Daily</li>
          </ul>
        </div>
      </div>

      <div class="row portfolio-container">
        <div class="col-lg-4 col-md-6 portfolio-item filter-monthly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-1.jpg" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>国内月度票房</h4>
              <div class="portfolio-links">
                <a href="#Modal1" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-monthly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-2.png" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>1-8月三项费用考核总体情况</h4>
              <div class="portfolio-links">
                <a href="#Modal2" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-monthly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-3.png" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>三项费用考核总体情况</h4>
              <div class="portfolio-links">
                <a href="#Modal3" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-yearly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-4.png" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>国内年度票房</h4>
              <div class="portfolio-links">
                <a href="#Modal4" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-daily">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-5.png" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>实时电影排片</h4>
              <div class="portfolio-links">
                <a href="#Modal5" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-yearly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-6.jpg" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>各省市票房占比</h4>
              <div class="portfolio-links">
                <a href="#Modal6" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-yearly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-7.png" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>热门电影评分分布</h4>
              <div class="portfolio-links">
                <a href="#Modal7" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-yearly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-8.jpg" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>各级城市票房占比</h4>
              <div class="portfolio-links">
                <a href="#Modal8" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>

        <div class="col-lg-4 col-md-6 portfolio-item filter-yearly">
          <div class="portfolio-wrap">
            <img src="/src/assets/img/portfolio/portfolio-9.jpg" class="img-fluid" alt="">
            <div class="portfolio-info">
              <h4>国产片、引进片占比</h4>
              <div class="portfolio-links">
                <a href="#Modal9" data-toggle="modal" title="App 1"><i class="icofont-eye"></i></a>
              </div>
            </div>
          </div>
        </div>


      </div>

    </div>
  </section>


  <!-- 图1 模态框-->
  <div class="portfolio-modal modal fade" id="Modal1" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>国内月度票房</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure1"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p>此图展示不同月份电影票房的变化情况，可见近年中国电影市场的发展趋势。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">折线图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">灯塔电影</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"> </a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图2-->
  <div class="portfolio-modal modal fade" id="Modal2" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>1-8月三项费用考核总体情况</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure2"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p>
            此图展示1-8月三项费用考核总体情况，可以清晰观察到各单位随时间变化，三项费用考核的结果。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">动态柱状图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">陕煤集团</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"> 2025.05.23</a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图3-->
  <div class="portfolio-modal modal fade" id="Modal3" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>三项费用考核总体情况</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure3"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p>
            此图展示不同月份电影人次的变化情况，可见近年中国电影市场的发展趋势。
          </p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">折线图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">灯塔电影</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"> </a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图4-->
  <div class="portfolio-modal modal fade" id="Modal4" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>国内年度票房</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure4"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p>此图展示不同年份电影票房的变化情况，可见近年中国电影市场的发展趋势。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">折柱混合</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">灯塔数据</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"> </a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图5-->
  <div class="portfolio-modal modal fade" id="Modal5" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>实时电影排片</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure5"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p id="figure5-info"></p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">饼图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">中国电影数据网</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"> </a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图6-->
  <div class="portfolio-modal modal fade" id="Modal6" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>各省市票房占比</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure6"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p id="figure6-info"><span class="ai-analysis"
              style="font-weight: bold;">AI分析：</span>2023年中国各省电影票房数据显示，广东省继续领跑，票房达到67.14亿元，显示其强大的电影市场消费能力。江苏、浙江紧随其后，形成第一梯队，票房均超过40亿元。四川、山东等省份也表现不俗，票房超过20亿元。值得注意的是，尽管部分西部和北部省份的票房相对较低，但也呈现出稳步增长的趋势。从人次和场次来看，广东、江苏等省份同样位居前列，表明这些地区的观影氛围浓厚，电影市场活跃。同时，一些省份如天津、海南等，虽然总票房不高，但场均观影人次和放映场次相对较高，说明这些地区的电影市场具有较大的潜力。
          </p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">热力图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">中国电影数据网</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"></a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图7-->
  <div class="portfolio-modal modal fade" id="Modal7" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>热门电影评分分布</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure7"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p id="figure7-info">此图为豆瓣top250电影中的评分分布，供从业者参考。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">柱状图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">豆瓣电影网</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"></a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图8-->
  <div class="portfolio-modal modal fade" id="Modal8" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>各级城市票房占比</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure8"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p id="figure8-info">此图为豆瓣top250电影中的评分分布，供从业者参考。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">堆叠柱状图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">灯塔电影数据</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"></a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>
  <!-- 图9-->
  <div class="portfolio-modal modal fade" id="Modal9" tabindex="-1" role="dialog">
    <!-- 关闭标签要在dialog标签里面，才能关闭 -->
    <div class="modal-dialog">
      <div class="modal-content">
        <!-- 头部 -->
        <div class="modal-header">
          <!-- 标题 -->
          <h2>国产片、引进片占比</h2>
          <!-- 关闭按钮 -->
          <div class="close-modal" data-dismiss="modal">
            <div class="lr">
              <div class="rl">
              </div>
            </div>
          </div>
        </div>

        <!-- 图表详细页面 -->
        <div class="modal-body">
          <div class="echarts-figure" id="echarts-figure9"></div>
          <!-- 图表详细介绍说明字段 -->
          <br>
          <p id="figure9-info">此图为豆瓣top250电影中的评分分布，供从业者参考。</p>
          <!-- 3个li标签，可删除，可补充信息 -->
          <ul class="list-inline item-details">
            <li>类型：
              <strong><a href="#">堆叠柱状图</a>
              </strong>
            </li>
            <li>数据来源：
              <strong><a href="#">灯塔电影数据</a>
              </strong>
            </li>
            <li>更新时间：
              <strong><a href="#" class="figure-time"></a>
              </strong>
            </li>
          </ul>
        </div>
      </div>
    </div>

  </div>

  <!-- <a href="#" class="back-to-top" v-if="showBackToTop" @click="scrollToTop"><i class="icofont-simple-up"></i></a> -->


</template>

<script setup>
import { onMounted, ref, onUnmounted } from 'vue';
// import * as echarts from 'echarts';
import { loadScript } from '@/utils/loadScripts';

const showBackToTop = ref(false);

const scrollToTop = () => {
  e.preventDefault(); // 阻止默认行为
  // window.scrollTo({
  //       top: 0,
  //       behavior: 'smooth'
  //     });
};

// Define the scroll handler
const handleScroll = (e) => {
  showBackToTop.value = e.target.scrollTop > 600;
};

onMounted(() => {
  window.addEventListener('scroll', handleScroll,true);

  // 异步加载JS文件
  (async () => {
    try {
      await loadScript('/src/assets/vendor/jquery/jquery.min.js');
      await loadScript('/src/assets/vendor/bootstrap/js/bootstrap.bundle.min.js');
      await loadScript('/src/assets/vendor/jquery.easing/jquery.easing.min.js');
      await loadScript('/src/assets/vendor/jquery-sticky/jquery.sticky.js');
      await loadScript('/src/assets/vendor/venobox/venobox.min.js');
      await loadScript('/src/assets/vendor/waypoints/jquery.waypoints.min.js');
      await loadScript('/src/assets/vendor/counterup/counterup.min.js');
      await loadScript('/src/assets/vendor/isotope-layout/isotope.pkgd.min.js');
      await loadScript('/src/assets/js/echarts.min.js');
      await loadScript('/src/assets/js/china.js');
      await loadScript('/src/assets/js/visual.js');

      // 在脚本加载完成后进行初始化
      initPage();
      drawBoxofficeTrend();
      drawFigure2();
      drawFigure3();
    } catch (error) {
      console.error('Failed to load scripts', error);
    }
  })();

});



// Clean up the event listener on component unmount
onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll);
});


</script>


<style scoped>
@import url('/src/assets/vendor/icofont/icofont.min.css');
@import url('/src/assets/vendor/boxicons/css/boxicons.min.css');
@import url('/src/assets/vendor/animate.css/animate.min.css');
@import url('/src/assets/vendor/venobox/venobox.css');
@import url('/src/assets/css/visual.css');
</style>