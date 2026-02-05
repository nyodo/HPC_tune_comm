import { createRouter, createWebHistory } from "vue-router";
import NProgress from "nprogress"; //进度条
import "nprogress/nprogress.css";

NProgress.configure({
  showSpinner: false, //通过将其设置为 false 来关闭加载微调器。
});

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: "/login",
      // redirect: "/main",
      hidden: true,
    },
    {
      path: "/login",
      name: "login",
      meta: { title: "登陆" },
      component: () => import("@/views/login/login.vue"),
      // hidden: true,
    },
    {
      path: "/register",
      name: "register",
      meta: { title: "注册" },
      component: () => import("@/views/login/register.vue"),
      // hidden: true,
    },
    {
      path: "/test",
      name: "test",
      meta: { title: "测试" },
      component: () => import("@/views/test.vue"),
    },
    {
      path: "/main",
      name: "main",
      meta: { title: "主页面" },
      component: () => import("@/views/main/main1.vue"),
      children: [
        {
          path: "dcu",
          name: "dcu",
          meta: { title: "DCU平台" },
          component: () => import("@/views/main/main-dcu.vue"),
        },
        {
          path: "mt-model",
          name: "mt-model",
          meta: { title: "迈创平台建模" },
          component: () => import("@/views/main/main-mt-model.vue"),
        },
        {
          path: "mt3000",
          name: "mt3000",
          meta: { title: "MT-3000平台" },
          component: () => import("@/views/main/main-mt3000.vue"),
        },
        {
          path: "mt-opt",
          name: "mt-opt",
          meta: { title: "迈创平台调优" },
          component: () => import("@/views/main/main-mt-opt.vue"),
        },
        {
          path: "llm",
          name: "llm",
          meta: { title: "大模型分析" },
          component: () => import("@/views/main/main-llm.vue"),
        }
      ],
    },
    {
      path: "/top",
      name: "top",
      meta: { title: "上导航栏" },
      component: () => import("@/views/main/main.vue"),
      children: [
        {
          path: "intro",
          name: "intro",
          meta: { title: "技术介绍" },
          component: () => import("@/views/topbar/top-intro.vue"),
        },
        {
          path: "message",
          name: "message",
          meta: { title: "联系我们" },
          component: () => import("@/views/topbar/top-message.vue"),
        }
      ],
    },
    {
      path: "/edit",
      name: "edit",
      meta: { title: "后台管理" },
      component: () => import("@/views/main/main.vue"),
      children: [
        {
          path: "e-user",
          name: "e-user",
          meta: { title: "用户记录" },
          component: () => import("@/views/edit/user.vue"),
        },
        {
          path: "e-dcu",
          name: "e-dcu",
          meta: { title: "DCU分析记录" },
          component: () => import("@/views/edit/dcu.vue"),
        },
        {
          path: "e-relation",
          name: "e-relation",
          meta: { title: "关系记录" },
          component: () => import("@/views/edit/relation.vue"),
        },
        {
          path: "e-session",
          name: "e-session",
          meta: { title: "会话记录" },
          component: () => import("@/views/edit/session.vue"),
        },
        {
          path: "e-qa",
          name: "e-qa",
          meta: { title: "问答记录" },
          component: () => import("@/views/edit/qa.vue"),
        },
        {
          path: "e-file",
          name: "e-file",
          meta: { title: "文件记录" },
          component: () => import("@/views/edit/file.vue"),
        }
      ],
    }
  ],
});

import defaultSettings from "@/settings";

//路由全局前置钩子
router.beforeEach((to, from, next) => {
  NProgress.start();
  document.title = `${to.meta.title || "首页"} - ${defaultSettings.title}`;
  
  // ===== 临时禁用登录验证，方便测试 =====
  next();
  return;
  // ===== 以下是原始登录验证代码 =====
  
  let token = localStorage.getItem("token");
  if (token) {
    next();
  } else {
    if (to.path === "/login" || to.path === "/register") {
      next();
    } else {
      next({
        path: "/login",
      });
    }
  }
});

// //路由全局后置钩子
router.afterEach(() => {
  NProgress.done();
});

export default router;
