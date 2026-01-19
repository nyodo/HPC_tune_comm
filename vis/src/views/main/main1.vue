<template>
  <el-container class="home-container">
    <!-- Header -->
    <el-header class="header">
      <div class="header-left">
        <img src="/src/assets/xjtu.svg" alt="Logo" class="logo">
        <span class="header-title">超算平台建模调优系统</span>
      </div>
      <div class="header-right">
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

    <!-- Main Content -->
    <el-main class="content">
      <!-- 当路径为 /main 时，显示架构选择页面 -->
      <template v-if="isMainPath">
        <div class="main-content-wrapper">
          <div class="architecture-container">
            <!-- DCU平台卡片 -->
            <el-card 
              class="architecture-card" 
              @click="showPlatformDrawer('dcu')"
            >
              <div class="card-header">
                <h3>DCU平台</h3>
              </div>
              <div class="card-body">
                <p>DCU平台支持多种高性能超算平台。</p>
                <el-button type="primary">查看平台</el-button>
              </div>
            </el-card>

            <!-- MT-3000平台卡片 -->
            <el-card 
              class="architecture-card" 
              @click="showPlatformDrawer('mt3000')"
            >
              <div class="card-header">
                <h3>MT-3000平台</h3>
              </div>
              <div class="card-body">
                <p>MT-3000平台上的超算任务支持多种建模和优化。</p>
                <el-button type="primary">查看平台</el-button>
              </div>
            </el-card>
          </div>
        </div>

        <!-- 平台列表抽屉 -->
        <el-drawer
          v-model="drawerVisible"
          :title="drawerTitle"
          :size="500"
          :with-header="true"
        >
          <div class="platform-drawer-content">
            <div v-if="selectedPlatform === 'dcu'" class="platform-list">
              <el-card 
                v-for="platform in dcuPlatforms" 
                :key="platform.id"
                class="platform-card" 
                @click="navigateToPlatformDetail(platform.route)"
              >
                <div class="platform-card-content">
                  <h3>{{ platform.name }}</h3>
                  <p v-if="platform.description">{{ platform.description }}</p>
                </div>
              </el-card>
            </div>

            <div v-if="selectedPlatform === 'mt3000'" class="platform-list">
              <el-card 
                v-for="platform in mt3000Platforms" 
                :key="platform.id"
                class="platform-card" 
                @click="navigateToPlatformDetail(platform.route)"
              >
                <div class="platform-card-content">
                  <h3>{{ platform.name }}</h3>
                  <p v-if="platform.description">{{ platform.description }}</p>
                </div>
              </el-card>
            </div>
          </div>
        </el-drawer>
      </template>
      
      <!-- 当路径为子路由时，显示子路由内容 -->
      <router-view v-else />
    </el-main>

    <!-- Footer -->
    <el-footer class="footer">
      <span>© 2025 超算平台建模调优系统</span>
    </el-footer>
  </el-container>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage } from 'element-plus';

const nickname = ref('');
const selectedPlatform = ref(''); // 当前选择的平台大类
const drawerVisible = ref(false); // 控制抽屉显示
const router = useRouter();
const route = useRoute();

// 判断当前路径是否为 /main（主路径，没有子路径）
const isMainPath = computed(() => {
  const path = route.path;
  // 检查路径是否正好是 /main 或 /main/，没有其他子路径
  return path === '/main' || path === '/main/';
});

// MT-3000平台下的平台列表
const mt3000Platforms = ref([
  {
    id: 'mt3000',
    name: 'MT3000',
    description: 'MT-3000超算平台',
    route: '/main/mt3000?platform=MT3000'
  },
  {
    id: 'mt5000',
    name: 'MT5000',
    description: 'MT-5000超算平台',
    route: '/main/mt-model?platform=MT5000'
  },
  {
    id: 'mt8000',
    name: 'MT8000',
    description: 'MT-8000超算平台',
    route: '/main/mt-model?platform=MT8000'
  }
]);

// DCU平台下的平台列表
const dcuPlatforms = ref([
  {
    id: 'dcu',
    name: 'DCU平台',
    description: '海光DCU超算平台',
    route: '/main/dcu'
  }
]);

onMounted(() => {
  nickname.value = localStorage.getItem('username') || '';
});

// 计算抽屉标题
const drawerTitle = computed(() => {
  if (selectedPlatform.value === 'dcu') {
    return 'DCU平台 - 平台列表';
  } else if (selectedPlatform.value === 'mt3000') {
    return 'MT-3000平台 - 平台列表';
  }
  return '平台列表';
});

// 用户点击架构卡片，显示平台列表抽屉
const showPlatformDrawer = (platform) => {
  selectedPlatform.value = platform;
  drawerVisible.value = true;
};

// 用户点击平台卡片，进入对应的平台详情页面
const navigateToPlatformDetail = (routePath) => {
  drawerVisible.value = false;
  router.push(routePath);
};

const quitLogin = () => {
  localStorage.clear();
  router.push('/login');
  ElMessage.success({
    message: '退出成功！',
    duration: 1500,
  });
};
</script>

<style scoped>
.home-container {
  height: 100vh;
  background-color: #f0f2f5;
  display: flex;
  flex-direction: column;
}

.header {
  height: 70px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  background-color: #003f88;
  color: white;
}

.header-left {
  display: flex;
  align-items: center;
}

.header-title {
  margin-left: 15px;
  font-size: 24px;
  font-weight: bold;
}

.header-right {
  display: flex;
  align-items: center;
}

.avatar {
  margin-right: 20px;
}

.main-content-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 200px);
  padding: 40px 20px;
}

.architecture-container {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 60px;
  flex-wrap: wrap;
}

.architecture-card {
  width: 380px;
  min-height: 300px;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
  border-radius: 16px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: 2px solid transparent;
  background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
  overflow: hidden;
  position: relative;
}

.architecture-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #003f88, #0056b3);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.architecture-card:hover::before {
  opacity: 1;
}

.architecture-card:hover {
  transform: translateY(-10px) scale(1.02);
  box-shadow: 0 16px 32px rgba(0, 63, 136, 0.25);
  border-color: #003f88;
}

.card-header {
  text-align: center;
  padding: 20px 0 10px 0;
}

.card-header h3 {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  color: #003f88;
}

.card-body {
  padding: 24px 20px 30px 20px;
  text-align: center;
}

.card-body p {
  margin: 24px 0 30px 0;
  color: #666;
  font-size: 15px;
  line-height: 1.8;
  min-height: 48px;
}

.card-body .el-button {
  padding: 12px 32px;
  font-size: 16px;
  font-weight: 500;
}

.platform-drawer-content {
  padding: 20px 0;
}

.platform-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.platform-card {
  width: 100%;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  transition: all 0.3s ease;
  border: 2px solid transparent;
}

.platform-card:hover {
  transform: translateX(5px);
  box-shadow: 0 4px 16px rgba(0, 63, 136, 0.2);
  border-color: #003f88;
  background-color: #f8f9fa;
}

.platform-card-content {
  padding: 24px;
}

.platform-card-content h3 {
  margin: 0 0 10px 0;
  font-size: 22px;
  font-weight: bold;
  color: #003f88;
}

.platform-card-content p {
  margin: 0;
  color: #666;
  font-size: 14px;
  line-height: 1.5;
}

.footer {
  text-align: center;
  padding: 20px;
  background-color: #003f88;
  color: white;
}
</style>
