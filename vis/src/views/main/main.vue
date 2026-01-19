<template>
  <el-container>
    <el-header class="header">
      <div class="header-left">
        <img src="/src/assets/xjtu.svg" alt="">
        <span>超算平台建模调优系统</span>
      </div>
      <div class="header-mid">
        <router-link to="/main/dcu-model"><span>主页</span></router-link>
        <!-- <router-link to="/top/video"><span>演示</span></router-link> -->
        <router-link to="/top/intro"><span>使用说明</span></router-link>
        <router-link to="/top/message"><span>联系我们</span></router-link>
      </div>
      <div class="header-right">
        <div class="refresh-icon" @click="refreshPage">
          <el-icon color="#fff" :size="24">
            <Refresh />
          </el-icon>
        </div>
        <el-dropdown class="dropdown">
          <div class="avatar">
            <el-avatar :size="26"
              src="https://gw.alipayobjects.com/zos/antfincdn/XAosXuNZyF/BiazfanxmamNRoxxVxka.png" />
            <span class="nickname">{{ nickname }}</span>
          </div>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item disabled>个人中心</el-dropdown-item>
              <el-dropdown-item disabled>个人设置</el-dropdown-item>
              <el-dropdown-item divided @click="quitLogin">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </el-header>
    <el-container class="content">
      <el-aside width="250px">
        <el-menu router :collapse="isCollapse" :default-active="$route.path" active-text-color="#003f88"
          background-color="#fff" text-color="#333" :collapse-transition="true">
          
            <el-menu-item index="/main/dcu-model">
              <el-icon>
                <ChatLineSquare />
              </el-icon>海光平台访存性能自动建模
            </el-menu-item>
            <el-menu-item index="/main/dcu-opt">
              <el-icon>
                <ChatLineSquare />
              </el-icon>海光平台访存性能自动调优
            </el-menu-item>
            <el-menu-item index="/main/mt-model">
              <el-icon>
                <ChatLineSquare />
              </el-icon>迈创平台访存性能自动建模
            </el-menu-item>
            <!-- <el-menu-item index="/main/mt-opt">
              <el-icon>
                <ChatLineSquare />
              </el-icon>迈创平台访存性能自动调优
            </el-menu-item> -->
            <!-- <el-menu-item index="/main/llm">
              <el-icon>
                <ChatLineSquare />
              </el-icon>大模型分析
            </el-menu-item> -->
            <!-- <el-menu-item index="/main/visual">
              <el-icon>
                <DataLine  />
              </el-icon>数据可视化
            </el-menu-item> -->
            <el-menu-item v-if="isAdmin == 1" index="/edit/e-user">
              <el-icon>
                <Edit />
              </el-icon>用户管理
            </el-menu-item>
            <el-menu-item v-if="isAdmin == 1" index="/edit/e-dcu">
              <el-icon>
                <Edit />
              </el-icon>DCU分析记录
            </el-menu-item>
            <!-- <el-menu-item v-if="isAdmin == 1" index="/edit/e-relation">
              <el-icon>
                <Edit />
              </el-icon>天河分析记录
            </el-menu-item> -->
            <!-- <el-menu-item v-if="isAdmin == 1" index="/edit/e-session">
              <el-icon>
                <Edit />
              </el-icon>会话记录
            </el-menu-item> -->
            <!-- <el-menu-item v-if="isAdmin == 1" index="/edit/e-qa">
              <el-icon>
                <Edit />
              </el-icon>问答记录
            </el-menu-item> -->
            <el-menu-item v-if="isAdmin == 1" index="/edit/e-file">
              <el-icon>
                <Edit />
              </el-icon>文件记录
            </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';

const isCollapse = ref(false);
const nickname = ref('');
const isAdmin = ref(false);//是否为管理员，可控制导航栏一些选项

const router = useRouter();

onMounted(() => {
  nickname.value = localStorage.getItem("nickname");
  isAdmin.value = localStorage.getItem("isAdmin");
  // console.log(isAdmin.value)
});

const quitLogin = () => {
  localStorage.clear();
  router.push('/login');
  ElMessage.success({
    message: '退出成功！',
    duration: 1500
  });
};

const refreshPage = () => {
  window.location.reload();
};
</script>

<style scoped>
.content {
  height: calc(100vh - 70px);
}

.el-aside {
  z-index: 10;
  box-shadow: 2px 0 6px rgba(0, 21, 41, 0.35);
}

.el-header {
  line-height: 70px;
  height: 70px;
  padding: 0 10px;
  background-color: #003f88;
  color: #fff;
  z-index: 9;
  box-shadow: 0 1px 4px rgba(0, 21, 41, 0.08);
}

.el-main {
  background-color: #f0f2f5;
  position: relative;
  overflow: auto;
  /* 添加这行 */
  flex: 1;
  /* 确保主内容区占据剩余空间 */
  padding:  0 ;
}


.el-menu {
  border: none;
  background-color: #fff;
}

.el-menu-item {
  /* background-color: #fff; */
  color: #474747 !important;
  /* 深灰色 */
}

.el-menu-item:hover {
  color: #003f88 !important;
  /* 鼠标悬停时蓝色 */
  background-color: #dedfe9;
}

.el-menu-item.is-active {
  background-color: #dedfe9 !important;
  /* 鼠标点击后的背景颜色 */
  color: #003f88 !important;
  /* 鼠标点击后的文字颜色 */
}

/* 头部导航栏 */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header-left {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 60px;
  line-height: 60px;
}

.header-left img {
  display: inline-block;
  height: 50px;
  width: 50px;
  margin-left: 40px;
  margin-right: 20px;
}

.header-mid {
  display: flex;
  align-items: center;
  justify-content: space-evenly;
  width: 500px;
  font-size: 16px;
}

.header-mid span {
  cursor: pointer;
  color: #fff;
}

.header-mid a {
  text-decoration: none;
  /* 消除下划线 */
}



.header-right {
  display: flex;
  align-items: center;
}

.refresh-icon {
  cursor: pointer;
  height: 100%;
  padding: 0 12px;
  display: flex;
  align-items: center;
}

.dropdown {
  cursor: pointer;
  height: 100%;
  display: flex;
  align-items: center;
  line-height: 60px;
}

.avatar {
  padding: 0 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.nickname {
  margin-left: 10px;
  color: #909399;
}
</style>
