
// 求和函数
function sum(numbers) {
  let total = 0;
  for (let number of numbers) {
    total += number;
  }
  return total;
}
// 二维数组转置函数
function transpose(matrix) {
  return matrix[0].map((_, colIndex) =>
    matrix.map(row => row[colIndex]));
}
// 实时模块-大盘趋势图
function drawBoxofficeTrend() {
  let dateList = [
    '1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月'
  ];

  let dataList = [
    14567, 15234, 14890, 15123, 14789, 15678, 14321, 15000
  ];
  let myChart = echarts.init(document.getElementById("daily-boxoffice-chart"));
  option = {
    color: ['#4871ae'],
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        label: {
          backgroundColor: '#6a7985'
        }
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: [
      {
        type: 'category',
        boundaryGap: false,
        data: dateList
      }
    ],
    yAxis: [
      {
        type: 'value'
      }
    ],
    series: [
      {
        name: '三项支出（万元）',
        type: 'line',
        smooth: true,
        lineStyle: {
          width: 4,
        },
        showSymbol: true,
        areaStyle: {
          opacity: 0.8,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            {
              offset: 0,
              color: '#4871ae'
            },
            {
              offset: 1,
              color: '#FFF'
            }
          ])
        },
        emphasis: {
          focus: 'series'
        },
        data: dataList
      },
    ]
  };
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
}
// 每月票房变化
function drawFigure1(monthly_data) {
  let dataList1 = monthly_data.monthly_data_2022.map(item => item[1])
  let dataList2 = monthly_data.monthly_data_2023.map(item => item[1])
  let dataList3 = [
    25.83,
    111.59,
    25.64
  ];
  // 基于准备好的dom，初始化echarts实例
  let myChart = echarts.init(document.getElementById("echarts-figure1"));
  let option = {
    tooltip: {
      trigger: "axis",
      axisPointer: {
        lineStyle: {
          color: "#dddc6b"
        }
      }
    },
    legend: {
      top: "0%",
      textStyle: {
        color: "rgba(0,0,0,.7)",
        fontSize: "16"
      }
    },

    xAxis: [
      {
        type: "category",
        boundaryGap: false,
        show: true,
        axisLabel: {
          textStyle: {
            fontSize: 14
          }
        },
        data: ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
      }
    ],

    yAxis: [
      {
        type: "value",
        axisTick: { show: true },
        axisLabel: {
          textStyle: {
            fontSize: 14
          },
          formatter: '{value} 亿'
        }
      }
    ],
    series: [
      {
        name: "2022年",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 5,
        showSymbol: false,
        lineStyle: {
          normal: {
            // color: "#0184d5",
            width: 2
          }
        },
        data: dataList1
      },
      {
        name: "2023年",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 5,
        showSymbol: false,
        lineStyle: {
          normal: {
            // color: "#00d887",
            width: 2
          }
        },
        data: dataList2
      },
      {
        name: "2025年",
        type: "line",
        smooth: true,
        symbol: "circle",
        symbolSize: 5,
        showSymbol: false,
        lineStyle: {
          normal: {
            width: 2
          }
        },
        data: dataList3
      }
    ]
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
}
// 每月场次变化
function drawFigure2() {
  // 基于准备好的dom，初始化echarts实例
  let myChart = echarts.init(document.getElementById("echarts-figure2"));

  const updateFrequency = 3000;
  const dimension = 0;
  const teamColors = {
    '综采一队': '#1f77b4', // 蓝色
    '综采二队': '#ff7f0e', // 橙色
    '掘进一队': '#2ca02c', // 绿色
    '掘进二队': '#d62728', // 红色
    '探放水队': '#9467bd', // 紫色
    '机电队': '#8c564b', // 棕色
    '胶带运输队': '#e377c2' // 粉色
  };


  // 随机生成数据
  function generateRandomData() {
    const teams = ['综采一队', '综采二队', '掘进一队', '掘进二队', '探放水队', '机电队', '胶带运输队'];
    const data = [];
    for (let month = 1; month <= 8; ++month) {
      for (let i = 0; i < teams.length; ++i) {
        data.push([Math.random() * 100, teams[i], month + '月']);
      }
    }
    return data;
  }

  const data = generateRandomData();
  const months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月'];

  $.when(

  ).done(function () {
    const startIndex = 0;
    const startMonth = months[startIndex];
    let isPaused = false;
    let timeoutHandles = [];

    const option = {
      grid: {
        top: 10,
        bottom: 60,  // 调整坐标轴底部位置
        left: 150,
        right: 80
      },
      toolbox: {
        feature: {
          dataView: { show: true, readOnly: false },
          magicType: { show: true, type: ['line', 'bar'] },
          restore: { show: true },
          saveAsImage: { show: true,
          pixelRatio: 6}
        }
      },
      xAxis: {
        max: 'dataMax',
        axisLabel: {
          formatter: function (n) {
            return Math.round(n) + '';
          }
        }
      },
      dataset: {
        source: data.filter(function (d) {
          return d[2] === startMonth;
        })
      },
      yAxis: {
        type: 'category',
        inverse: true,
        max: 7,
        axisLabel: {
          show: true,
          fontSize: 14
        },
        animationDuration: 300,
        animationDurationUpdate: 300
      },
      series: [
        {
          realtimeSort: true,
          seriesLayoutBy: 'column',
          type: 'bar',
          itemStyle: {
            color: function (param) {
              return teamColors[param.value[1]] || '#5470c6';
            }
          },
          encode: {
            x: dimension,
            y: 1
          },
          label: {
            show: true,
            precision: 1,
            position: 'right',
            valueAnimation: true,
            fontFamily: 'monospace'
          }
        }
      ],
      // Disable init animation.
      animationDuration: 0,
      animationDurationUpdate: updateFrequency,
      animationEasing: 'linear',
      animationEasingUpdate: 'linear',
      graphic: [
        {
          type: 'text',
          right: 160,
          bottom: 80,
          style: {
            text: startMonth,
            font: 'bolder 80px monospace',
            fill: 'rgba(100, 100, 100, 0.25)'
          },
          z: 100
        },
        {
          type: 'rect',
          left: 0,
          bottom: 80,
          shape: {
            width: 60, // 稍微加宽一点  
            height: 30,
            cornerRadius: 5 // 添加圆角  
          },
          style: {
            fill: '#5470c6', // 使用较深的蓝色  
            stroke: '#3450a6', // 使用较深的边框色  
            lineWidth: 2,
            shadowBlur: 10, // 添加阴影效果  
            shadowColor: 'rgba(0, 0, 0, 0.3)' // 阴影颜色  
          },
          onclick: function () {
            if (isPaused) {
              play();
            } else {
              pause();
            }
          }
        },
        {
          type: 'text',
          left: 13,
          bottom: 86,
          style: {
            text: '暂停',
            font: 'bold 18px Arial', // 使用更清晰的字体和稍大的字号  
            fill: '#fff',
            textAlign: 'center' // 文本居中  
          },
          onclick: function () {
            if (isPaused) {
              play();
            } else {
              pause();
            }
          }
        },
        {
          type: 'rect',
          left: 0,
          bottom: 40,
          shape: {
            width: 60, // 稍微加宽一点  
            height: 30,
            cornerRadius: 5 // 添加圆角  
          },
          style: {
            fill: '#5470c6', // 使用较深的蓝色  
            stroke: '#3450a6', // 使用较深的边框色  
            lineWidth: 2,
            shadowBlur: 10, // 添加阴影效果  
            shadowColor: 'rgba(0, 0, 0, 0.3)' // 阴影颜色  
          },
          onclick: function () {
            restart();
          }
        },
        {
          type: 'text',
          left: 13,
          bottom: 46,
          style: {
            text: '重播',
            font: 'bold 18px Arial', // 使用更清晰的字体和稍大的字号  
            fill: '#fff',
            textAlign: 'center' // 文本居中  
          },
          onclick: function () {
            restart();
          }
        }
      ]
    };

    myChart.setOption(option);

    function scheduleUpdates() {
      updateMonth(months[0]);
      for (let i = startIndex; i < months.length - 1; ++i) {
        (function (i) {
          const handle = setTimeout(function () {
            if (!isPaused) {
              updateMonth(months[i + 1]);
            }
          }, (i - startIndex) * updateFrequency);
          timeoutHandles.push(handle);
        })(i);
      }
    }

    function updateMonth(month) {
      const source = data.filter(function (d) {
        return d[2] === month;
      });
      // console.log(month);
      option.series[0].data = source;
      option.graphic[0].style.text = month;
      myChart.setOption(option);
    }

    function play() {
      isPaused = false;
      option.graphic[2].style.text = '暂停';
      myChart.setOption(option);
      scheduleUpdates();
    }

    function pause() {
      isPaused = true;
      option.graphic[2].style.text = '播放';
      myChart.setOption(option);
      timeoutHandles.forEach(clearTimeout);
      timeoutHandles = [];
    }

    function restart() {
      isPaused = false;
      timeoutHandles.forEach(clearTimeout);
      timeoutHandles = [];
      option.graphic[0].style.text = startMonth;
      myChart.setOption(option);
      scheduleUpdates();
    }
      // 初始化时显示第一个月份

    scheduleUpdates();
  });



};
// 每月人次变化
function drawFigure3() {
  // 基于准备好的dom，初始化echarts实例
  let myChart = echarts.init(document.getElementById("echarts-figure3"));
  let option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#999'
        }
      }
    },
    toolbox: {
      feature: {
        dataView: { show: true, readOnly: false },
        magicType: { show: true, type: ['line', 'bar'] },
        restore: { show: true },
        saveAsImage: { show: true,
        pixelRatio: 6}
      }
    },
      legend: {
      data: ['计划', '支出', '节超率', '奖罚'],
      top: 10 // 图例放置在顶部
    },
    xAxis: [
      {
        type: 'category',
        data: ['综采一队', '综采二队', '掘进一队', '掘进二队', '探放水队', '机电队', '胶带运输队', '通风维护区'],
        axisPointer: {
          type: 'shadow'
        }
      }
    ],
    yAxis: [
      {
        type: 'value',
        name: '费用',
        min: 0,
        max: 500,
        interval: 100,
        axisLabel: {
          formatter: '{value} 万元'
        }
      },
      {
        type: 'value',
        name: '节超率',
        min: -10,
        max: 10,
        interval: 2,
        axisLabel: {
          formatter: '{value} %'
        }
      }
    ],
    series: [
      {
        name: '计划',
        type: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value + ' 万元';
          }
        },
        data: [273.64, 315.79, 133.82, 110.22, 182.46, 475.13, 429.07, 114.98]
      },
      {
        name: '支出',
        type: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value + ' 万元';
          }
        },
        data: [256.45, 303.04, 142.41, 109.59, 185.63, 458.64, 409.50, 105.85]
      },
      {
        name: '节超率',
        type: 'line',
        yAxisIndex: 1,
        tooltip: {
          valueFormatter: function (value) {
            return value + ' %';
          }
        },
        data: [6.28, 4.04, -6.42, 0.57, -1.73, 3.47, 4.56, 7.94]
      },
      {
        name: '奖罚',
        type: 'bar',
        tooltip: {
          valueFormatter: function (value) {
            return value + ' 万元';
          }
        },
        data: [7.79, 2.49, -3.92, -1.48, 0.24, 7.52, 8.81, 7.58]
      }
    ]
  };
  
  
  

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 每年总票房变化
function drawFigure4(yearly_data) {
  dataList1 = yearly_data.map(item => item[0])
  dataList2 = yearly_data.map(item => item[1])
  dataList3 = yearly_data.map(item => Math.round(item[2] * 10000) / 100)
  // 基于准备好的dom，初始化echarts实例
  let myChart = echarts.init(document.getElementById("echarts-figure4"));
  let option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        crossStyle: {
          color: '#999'
        }
      }
    },
    legend: {
      show: true
    },
    xAxis: [
      {
        type: 'category',
        data: dataList1,
        axisPointer: {
          type: 'shadow'
        }
      }
    ],
    yAxis: [
      {
        type: 'value',
        name: '总票房',
        min: 0,
        max: 750,
        interval: 150,
        axisLabel: {
          formatter: '{value}亿'
        }
      },
      {
        type: 'value',
        name: '比例',
        min: -100,
        max: 150,
        // interval: 50,
        axisLabel: {
          formatter: '{value}%'
        }
      }
    ],
    series: [
      {
        name: '年度票房',
        type: 'bar',
        color: 'skyblue',
        tooltip: {
          valueFormatter: function (value) {
            return value + '亿';
          }
        },
        data: dataList2
      },
      {
        name: '增长比例',
        type: 'line',
        color: 'orange',
        yAxisIndex: 1,
        tooltip: {
          valueFormatter: function (value) {
            return value + '%';
          }
        },
        data: dataList3
      }
    ]
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 实时排片
function drawFigure5(sessionList) {
  $("#figure5-info").text("此图是" + sessionList[0][1] + "，" + sessionList[0][2] + "的电影排片，总场次为" + sessionList[0][3] + "场。")
  let dataList = []
  sessionList.forEach(item => {
    let dict = {}
    let index = item[5].indexOf('%');
    dict['value'] = item[5].substring(0, index);
    dict['name'] = item[4]
    dataList.push(dict)
  })
  let myChart = echarts.init(document.getElementById("echarts-figure5"));
  let option = {
    tooltip: {
      trigger: 'item',
      textStyle: {
        align: 'center'
      },
      formatter: function (params) {
        return (
          '排片比例' +
          '<br/>' +
          params.marker +
          params.name +
          '&nbsp;&nbsp;&nbsp;' +
          params.value +
          '%'
        );
      }
    },
    series: [
      {
        name: '排片比例',
        type: 'pie',
        radius: '70%',
        data: dataList,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 各省票房占比，中国地图热力图
function drawFigure6(monthly_data) {
  let data_2023 = monthly_data.province_data_2023;
  let data_list = []
  data_2023.forEach(item => {
    let dict = {}
    dict['name'] = item[1]
    dict['value'] = item[2]
    data_list.push(dict)
  })

  let data_value_list = data_list.map(item => item['value'])
  let sum_boxoffice = sum(data_value_list)
  max1_name = data_2023[0][1]
  max1_percent = (data_2023[0][2] / sum_boxoffice * 100).toFixed(2)
  max2_name = data_2023[1][1]
  max2_percent = (data_2023[1][2] / sum_boxoffice * 100).toFixed(2)
  // $("#figure6-info").text(`此图是${data_2023[0][0]}年各省总票房占比(不包含港澳台)，${max1_name}、${max2_name}省占比最高，分别为${max1_percent}%、${max2_percent}%`)

  let myChart = echarts.init(document.getElementById("echarts-figure6"));
  let option = {
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        let num
        let showHtml = ''
        let percentage
        if (isNaN(params.value)) {
          num = '未统计'
          return
        } else {
          num = params.value
          percentage = (num / sum(data_value_list) * 100).toFixed(2)
          percentage += '%'
        }
        showHtml += `
                    <span style="display: flex;">
                        ${'省份'}：${params.name}</br>
                        ${'票房(亿)'}：${num}</br>
                        ${'占比'}：${percentage}
                    </span>
                `
        return showHtml

      }
    },

    dataRange: {
      x: 'left',
      y: 'bottom',
      min: 0,
      max: Math.max.apply(null, data_value_list),
      text: ['高', '低'], // 文本，默认为数值文本
      calculable: true,
      inRange: {
        color: ['#fff', '#1989fa'],
      }
    },
    series: [{
      name: '数据',
      type: 'map',
      mapType: 'china',
      roam: false,
      selectedMode: false,
      itemStyle: {
        normal: {
          label: {
            show: true,
            textStyle: {
              color: 'black'
            }
          }
        },
        emphasis: {
          areaColor: '#95ec69',
          label: {
            show: true

          }
        }
      },
      data: data_list
    }]
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 热门电影评分分布
function drawFigure7(score_data) {
  score_list = score_data.score.map(item => item * 1.0);
  let min_score = Math.min.apply(null, score_list);
  $("#figure7-info").text(`此图为豆瓣top250电影中的评分分布，${min_score}分以上可以评为热门影片。`)
  let myChart = echarts.init(document.getElementById("echarts-figure7"));
  let option = {
    color: ['#3398DB'],
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: function (params) {
        return (
          params[0].name +
          ' 分<br/>' +
          params[0].marker +
          '&nbsp;' +
          params[0].value +
          ' 部'
        );
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: score_data.score
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: '{value}部'
      }
    },
    series: [
      {
        data: score_data.num,
        type: 'bar',
        name: '电影数量',
        barWidth: '60%'
      }
    ]
  };

  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 各级城市占比
function drawFigure8(city_boxoffice_ratio) {
  $("#figure8-info").text(`此图为历年各级城市票房占比，如图所示，二、四线城市占比较高，四线城市占比逐年稳步增长。`)
  let myChart = echarts.init(document.getElementById("echarts-figure8"));
  const rawData = transpose(city_boxoffice_ratio);
  const totalData = [];
  for (let i = 0; i < rawData[0].length; ++i) {
    let sum = 0;
    for (let j = 0; j < rawData.length; ++j) {
      sum += rawData[j][i];
    }
    totalData.push(sum);
  }
  let option = {
    legend: {
      selectedMode: false
    },
    tooltip: {
      //显示 一组数据 当前拐点的数据
      trigger: "axis",
      triggerOn: 'mousemove',
      show: true
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: (value, index) => (value * 100).toFixed(1) + '%'
      }

    },
    xAxis: {
      type: 'category',
      data: [
        '2014',
        '2015',
        '2016',
        '2017',
        '2018',
        '2019',
        '2020',
        '2021',
        '2022',
        '2023'
      ]
    },
    series: [
      {
        name: '一线城市',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[0].map(item => item / totalData[0]),
        label: {
          show: false,
          formatter: (params) => Math.round(params.value * 1000) / 10 + '%'
        },
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(2) + '%'
        },
      },
      {
        name: '二线城市',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[1].map(item => item / totalData[1]),
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(2) + '%'
        }
      },
      {
        name: '三线城市',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[2].map(item => item / totalData[2]),
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(1) + '%'
        }
      },
      {
        name: '四线城市',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[3].map(item => item / totalData[3]),
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(1) + '%'
        }
      }
    ]
  };
  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};
// 国产引进占比
function drawFigure9(domestic_boxoffice_ratio) {
  $("#figure9-info").text(`此图为历年国产片、引进片票房占比，如图所示，国产片的地位逐年稳步增长。`)
  let myChart = echarts.init(document.getElementById("echarts-figure9"));
  const rawData = transpose(domestic_boxoffice_ratio);
  const totalData = [];
  for (let i = 0; i < rawData[0].length; ++i) {
    let sum = 0;
    for (let j = 0; j < rawData.length; ++j) {
      sum += rawData[j][i];
    }
    totalData.push(sum);
  }
  let option = {
    legend: {
      selectedMode: false
    },
    tooltip: {
      //显示 一组数据 当前拐点的数据
      trigger: "axis",
      triggerOn: 'mousemove',
      show: true
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        formatter: (value, index) => (value * 100).toFixed(1) + '%'
      }

    },
    xAxis: {
      type: 'category',
      data: [
        '2014',
        '2015',
        '2016',
        '2017',
        '2018',
        '2019',
        '2020',
        '2021',
        '2022',
        '2023'
      ]
    },
    series: [
      {
        name: '国产片',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[0].map(item => item / totalData[0]),
        label: {
          show: false,
          formatter: (params) => Math.round(params.value * 1000) / 10 + '%'
        },
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(1) + '%'
        },
      },
      {
        name: '引进片',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[1].map(item => item / totalData[1]),
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(1) + '%'
        }
      },
      {
        name: '其他',
        type: 'bar',
        stack: 'total',
        barWidth: '60%',
        data: rawData[2].map(item => item / totalData[2]),
        tooltip: {
          valueFormatter: (value) => (value * 100).toFixed(2) + '%'
        }
      }
    ]
  };
  // 使用刚指定的配置项和数据显示图表。
  myChart.setOption(option);
  window.addEventListener("resize", function () {
    myChart.resize();
  });
};

function initPage() {
  // 网页一些设置初始化
  $(document).ready(function () {
    // 图表第三行文字更新时间
    let today = new Date();
    // 获取年、月和日
    let year = today.getFullYear();
    let month = (today.getMonth() + 1).toString().padStart(2, '0'); // 月份从0开始，因此需要+1，然后使用padStart补零
    let day = today.getDate().toString().padStart(2, '0'); // 使用padStart补零
    // 将年、月、日拼接成所需的格式
    let formattedDate = year + '-' + month + '-' + day;
    $(".figure-time").text(formattedDate);

    // 实现实时数据标题右边的选择按钮
    $(".realtime .filter-list li").addClass("hover-effect");
    $(".realtime .filter-list li:first-child").css("background-color", "#1a3b6f");
    $(".realtime .filter-list li:first-child").removeClass("hover-effect");
    $(".cost-table").eq(1).css("display", "none");
    $(".cost-table").eq(2).css("display", "none");
    $(".cost-table").eq(3).css("display", "none");
    // 点击按钮事件：切换实时数据表格
    $(".realtime .filter-list li").click(function () {
      $(".realtime .filter-list li").addClass("hover-effect");
      $(".realtime .filter-list li").css("background-color", "")
      $(this).css("background-color", "#1a3b6f");
      $(this).removeClass("hover-effect");
      $(".cost-table").hide();
      switch ($(this).find("a").text()) {
        case "材料费":
          $("#table1").show();
          break;
        case "电费":
          $("#table2").show();
          break;
        case "辅助运输费":
          $("#table3").show();
          break;
        case "三项费用":
          $("#table4").show();
          break;
        default:
          console.log("实时数据按钮出错")
      }
    });
  });


  // Initiate the venobox plugin
  $('.venobox').venobox();
  // jQuery counterUp
  $('[data-toggle="counter-up"]').counterUp({
    delay: 10,
    time: 1000
  });

  // Portfolio isotope and filter

  let portfolioIsotope = $('.portfolio-container').isotope({
    itemSelector: '.portfolio-item',
    layoutMode: 'fitRows'
  });

  $('#portfolio-flters li').on('click', function () {
    $("#portfolio-flters li").removeClass('filter-active');
    $(this).addClass('filter-active');

    portfolioIsotope.isotope({
      filter: $(this).data('filter')
    });
  });

}

