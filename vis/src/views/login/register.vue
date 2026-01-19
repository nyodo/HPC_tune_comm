<template>
    <div class="login-container">
        <el-card class="box-card">
            <div class="login-body">
                <div class="login-title">注册</div>
                <el-form ref="form" :model="userForm">
                    <el-input placeholder="账号..." v-model="userForm.accountNumber" class="login-input">
                    </el-input>
                    <el-input placeholder="密码..." v-model="userForm.userPassword" class="login-input"
                        @keyup.enter.native="login" show-password>
                    </el-input>
                    <el-input placeholder="姓名..." v-model="userForm.name" class="login-input">
                    </el-input>
                    <el-input placeholder="手机号..." v-model="userForm.tel" class="login-input">
                    </el-input>
                    <!-- <el-select class="login-input" v-model="userForm.identity" clearable placeholder="请选择">
                        <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value">
                        </el-option>
                    </el-select> -->

                    <div class="login-submit">
                        <el-button type="primary" @click="$router.push('/login')">登录</el-button>
                        <el-button type="warning" autocomplete="off" @click="zhuce()">注册</el-button>
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
                accountNumber: '',
                userPassword: '',
                name: '',
                tel: '',
                identity: '0'
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
        zhuce() {
            // 检查用户名和密码是否为空  
            if (!this.userForm.accountNumber || !this.userForm.userPassword) {
                // 弹出错误警告  
                this.$message.error("用户名和密码不能为空！");
                return; // 停止进一步的操作  
            }
            this.$request.post("/register/", {
                username: this.userForm.accountNumber,
                password: this.userForm.userPassword,
                name: this.userForm.name,
                tel: this.userForm.tel,
                identity: this.userForm.identity
            }).then(res => {
                // console.log(res);
                if (res.data.meta.status === 200) {
                    this.$message.success("注册成功")
                    this.$router.push('/login')
                } else {
                    this.$message.error(res.data.meta.message);
                }
            });
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