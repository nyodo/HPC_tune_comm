<template>
    <div class="login-container">
        <el-card class="box-card">
            <div class="login-body">
                <div class="login-title">超算平台建模调优系统</div>
                <el-form ref="form" :model="userForm">
                    <el-input placeholder="请输入账号..." v-model="userForm.accountNumber" class="login-input">

                    </el-input>
                    <el-input placeholder="请输入密码..." v-model="userForm.userPassword" class="login-input"
                        @keyup.enter.native="login" show-password>

                    </el-input>

                    <el-select v-model="userForm.value" placeholder="请选择" class="login-input">
                        <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value">
                        </el-option>
                    </el-select>

                    <div class="login-submit">
                        <el-button type="primary" @click="login">登录</el-button>
                        <el-button type="warning" @click="goToRegister" style="margin-left: 20px">注册</el-button>
                    </div>
                    <!-- <div class="other-submit">
                      <router-link to="/login-admin" class="sign-in-text">管理员登录</router-link>
                  </div> -->
                </el-form>
            </div>
        </el-card>
    </div>
</template>

<script>
export default {
    name: "login",
    data() {
        return {
            userForm: {
                accountNumber: 'admin',
                userPassword: 'admin',
                value: ''
            },
            options: [
                {
                    value: '0',
                    label: '用户'
                },
                {
                    value: '1',
                    label: '管理员'
                }],
        };
    },

    methods: {
        login() {
            // 检查用户名和密码是否为空
            if (!this.userForm.accountNumber || !this.userForm.userPassword) {
                this.$message.error('用户名和密码不能为空！');
                return;
            }

            // 检查管理员选择是否为空
            if (!this.userForm.value) {
                this.$message.error('管理员选择不能为空！');
                return;
            }
            this.$request.post("/login/", {
                username: this.userForm.accountNumber,
                password: this.userForm.userPassword,
                value: this.userForm.value
            }).then(res => {
                if (!res || !res.data) {
                    this.$message.error('登录失败：后端无响应');
                    return;
                }

                console.log(res);
                if (res.data.meta.status === 200) {
                    console.log('login response raw:', res);
                    console.log(this.userForm.value, "后台回复", res.data.data.isAdmin)
                    localStorage.setItem("username", res.data.data.username);
                    localStorage.setItem("userId", res.data.data.user_id);
                    // localStorage.setItem("img_url_touxiang", res.data.data.img_url);
                    // localStorage.setItem("jianjie", res.data.data.jianjie);
                    localStorage.setItem("isAdmin", res.data.data.isAdmin);
                    localStorage.setItem("token", res.data.data.token);
                    localStorage.setItem('currentSessionId', -1)
                    this.$message.success({
                        message: '登录成功！',
                        duration: 1500
                    })
                    this.$router.push('/main')
                } else {
                    this.$message.error(res.data.meta.message);
                }
            }).catch(() => {
                // 统一在 request.js 拦截器里弹了错误，这里避免 Uncaught
            });
        },
        goToRegister() {
            this.$router.push('/register');
        },
        toIndex() {
            this.$router.replace({ path: '/index' });
        }
    }
}
</script>

<style scoped>
.login-container {
    background-image: url('/src/assets/bg.jpg');
    /* 替换为你的图片路径 */
    background-size: cover;
    /* 背景图片覆盖整个元素 */
    background-position: center -123px;
    /* 背景图片居中 */
    background-repeat: no-repeat;
    /* 不重复背景图片 */
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    width: 100%;

}

.login-body {
    padding: 30px;
    padding-left: 13px;
    width: 300px;
    height: 100%;
}

.login-title {
    padding-bottom: 30px;
    text-align: center;
    font-weight: 600;
    font-size: 20px;
    color: #409EFF;
    cursor: pointer;
}

.login-input {
    width: 280px;
    margin-bottom: 20px;
}

.login-submit {
    margin-top: 20px;
    display: flex;
    justify-content: center;
}

.sign-in-container {
    padding: 0 10px;
}

.sign-in-text {
    color: #409EFF;
    font-size: 16px;
    text-decoration: none;
    line-height: 28px;
}

.other-submit {
    display: flex;
    justify-content: space-between;
    margin-top: 30px;
    margin-left: 200px;
}
</style>