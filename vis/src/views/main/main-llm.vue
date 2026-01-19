<template>
    <div class="chatbot">
        <aside class="sidebar">
            <div class="icon-button" @click="newSession">
                <el-tooltip content="新建会话" placement="right" :hide-after="50">
                    <i class="mdi mdi-invoice-text-plus-outline"></i>
                </el-tooltip>
            </div>
            <div class="icon-button" @click="showHistory">
                <el-tooltip content="历史会话" placement="right" :hide-after="50">
                    <i class="mdi mdi-history"></i>
                </el-tooltip>
            </div>
            <div class="icon-button" @click="showCallword">
                <el-tooltip content="设置提示词" placement="right" :hide-after="50">
                    <i class="mdi mdi-tooltip-text-outline"></i>
                </el-tooltip>
            </div>
            <div class="icon-button" @click="showParams">
                <el-tooltip content="设置方法与参数" placement="right" :hide-after="50">
                    <i class="mdi mdi-cog"></i>
                </el-tooltip>
            </div>
        </aside>
        <div class="header">
            <el-dropdown @command="handleTaskType" trigger="click">
                <el-button type="primary" class="custom-button">
                    任务类型：{{ taskType }}<el-icon class="el-icon--right"><arrow-down /></el-icon>
                </el-button>
                <template #dropdown>
                    <el-dropdown-menu>
                        <el-dropdown-item command="ner">
                            NER
                        </el-dropdown-item>
                        <el-dropdown-item command="re">
                            RE
                        </el-dropdown-item>
                        <el-dropdown-item command="dialog">
                            QA
                        </el-dropdown-item>
                    </el-dropdown-menu>
                </template>
            </el-dropdown>
            <el-dropdown @command="handleCommand" trigger="click">
                <el-button type="primary" class="custom-button">
                    {{ sessionName }}<el-icon class="el-icon--right"><arrow-down /></el-icon>
                </el-button>
                <template #dropdown>
                    <el-dropdown-menu>
                        <el-dropdown-item command="edit">
                            <Edit style="width: 1.5em; height: 1.5em;" />修改名称
                        </el-dropdown-item>
                        <el-dropdown-item command="delete" style="color: red;">
                            <Delete style="width: 1.5em; height: 1.5em;" />删除
                        </el-dropdown-item>
                    </el-dropdown-menu>
                </template>
            </el-dropdown>
            <el-dropdown>
            </el-dropdown>
        </div>
        <div class="main">
            <div class="conversation" ref="conversation">
                <div class="contain" v-for="(message, index) in messages" :key="index">
                    <div class="message" :class="{ 'from-user': message.fromUser }">
                        <span v-html="renderMarkdown(message.text)"></span>
                    </div>
                    <div class="message file-message" v-if="message.filename">
                        <span>{{ message.filename }}</span>
                    </div>
                </div>
                <div id="endOfMessages" style="width: 423px;height: 10px;"></div>
            </div>
            <div class="filelist-tag">
                <el-tag v-for="file in files" :key="file.index" closable :disable-transitions="false"
                    @close="handleFileTagClose(file)">
                    {{ file.filename }}
                </el-tag>
            </div>
            <div class="input-box">
                <el-input class="input-text" @keyup.enter="sendMessage" v-model="newMessage" :autosize="{ minRows: 3 }"
                    type="textarea" placeholder="给大模型输入文本或文件" :disabled="isInputDisabled"/>
                <div class="icon-button upfile-button" @click="showUpload">
                    <el-tooltip content="上传文件" placement="top" :hide-after="50">
                    <i class="mdi mdi-link-variant"></i>
                </el-tooltip>
                </div>
                <div class="icon-button" @click="sendMessage">
                    <i class="mdi mdi-arrow-up-circle"></i>
                </div>
            </div>
        </div>
        <!-- 历史会话对话框 -->
        <el-dialog title="历史会话" v-model="historyDialogVisible" width="30%">
            <el-menu @select="selectHistorySession" class="history-menu">
                <el-menu-item v-for="(session, index) in historySessions" :key="index" :index="index.toString()">
                    <div style="font-weight: bold; margin-right: 5px;">{{ session.name }}</div>
                    <div>{{ session.firstMessage }}</div>
                    <div>{{ session.date }}</div>
                </el-menu-item>
            </el-menu>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="historyDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="confirmSelection">确定</el-button>
                </div>
            </template>
        </el-dialog>
        <!-- 设置提示词对话框 -->
        <el-dialog title="设置提示词" v-model="callwordDialogVisible" width="30%">
            <el-form :model="dailog_callword" label-width="140px" label-position="left">
                <el-form-item label="类别：">
                    <!-- <el-input v-model="dailog_callword.type" class="setting-input"></el-input> -->
                    <TagInput  v-model="dailog_callword.type" :test="testValue"/>
                </el-form-item>
                <el-form-item label="类别定义：">
                    <!-- <el-input v-model="dailog_callword.definition" class="setting-input" type="textarea"
                        placeholder="Please input"></el-input> -->
                    <TagInput v-model="dailog_callword.definition" :test="testValue"/>
                </el-form-item>
                <el-form-item label="示例：">
                    <TagInput v-model="dailog_callword.example" :test="testValue"/>
                </el-form-item>
            </el-form>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="callwordDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="saveCallword">确定</el-button>
                </div>
            </template>
        </el-dialog>
        <!-- 设置方法参数对话框 -->
        <el-dialog title="设置方法参数" v-model="paramsDialogVisible" width="30%">
            <el-form :model="dailog_params" label-width="120px" label-position="left">
                <el-form-item label="方法：">
                    <el-checkbox-group v-model="dailog_params.solution" @change="handleParamsChange">
                        <el-checkbox label="分词" size="large" />
                        <el-checkbox label="思维链" size="large" />
                        <el-checkbox label="多数投票" size="large" />
                        <!-- <el-checkbox label="自我验证" size="large" /> -->
                    </el-checkbox-group>
                </el-form-item>
                <!-- <el-form-item label="temperature：">
                    <el-slider v-model="dailog_params.param1" :step="0.1" :max="2" class="setting-input"
                        @change="handleParamsChange" />
                </el-form-item> -->
                <el-form-item label="采用次数：">
                    <el-input-number v-model="dailog_params.param2" :min="0" :max="1000" class="setting-input"
                        @change="handleParamsChange" />
                </el-form-item>
            </el-form>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="paramsDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="confirmParams">确定</el-button>
                </div>
            </template>
        </el-dialog>
        <!-- 上传文件对话框 -->
        <el-dialog title="上传文件" v-model="uploadDialogVisible" width="30%">
            <el-upload class="upload-demo" drag :action="uploadFileUrl" :limit="1"
                accept=".json,.txt" :on-success="handleSuccess" :on-error="handleError" :on-remove="handleRemove"
                ref="uploadRef" :file-list="fileList">
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                    拖拽或 <em>点击上传</em>
                </div>
                <template #tip>
                    <div class="el-upload__tip">
                        一次提问限制一个json或txt文件
                    </div>
                </template>
            </el-upload>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="uploadDialogVisible = false">取消</el-button>
                    <el-button type="primary" @click="uploadDialogVisible = false">确定</el-button>
                </div>
            </template>
        </el-dialog>
        <!-- Edit Dialog -->
        <el-dialog v-model="editDialogVisible" title="重命名" @close="handleEditDialogClose" width="30%">
            <el-input v-model="newName" placeholder="输入新名称"></el-input>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="handleEditDialogClose">取消</el-button>
                    <el-button type="primary" @click="confirmEdit">确认</el-button>
                </div>
            </template>
        </el-dialog>

        <!-- Delete Dialog -->
        <el-dialog v-model="deleteDialogVisible" title="永久删除会话" @close="handleDeleteDialogClose" width="30%">
            <span>本条会话数据将被永久删除，不可恢复及撤销。确定要删除吗？</span>
            <template #footer>
                <div class="dialog-footer">
                    <el-button @click="handleDeleteDialogClose">取消</el-button>
                    <el-button type="primary" @click="confirmDelete">确认</el-button>
                </div>
            </template>
        </el-dialog>
    </div>
</template>

<script setup>
import { ref,reactive, nextTick, onMounted,watch } from 'vue';
import '@mdi/font/css/materialdesignicons.css';
import axios from 'axios';
import { ElMessage } from 'element-plus';
import TagInput from '/src/components/TagInput.vue'
import MarkdownIt from 'markdown-it'

const uploadFileUrl = ref(import.meta.env.VITE_URL+'/upload-file');
// 处理Markdown字符串
const md = new MarkdownIt()
const renderMarkdown = (text) => {
    return md.render(text)
}

let selectedSessionIndex;
const historyDialogVisible = ref(false);
const historySessions = ref([]);
function showHistory() {
    getHistorySessionList();
    historyDialogVisible.value = true;
}
function selectHistorySession(index) {
    console.log('index:', index);
    selectedSessionIndex = index;
}
function confirmSelection() {
    if (selectedSessionIndex !== null) {
        const session = historySessions.value[selectedSessionIndex];
        // console.log('选择的会话:', session);
        getSession(session.session_id);
    }
    historyDialogVisible.value = false;
}
async function getHistorySessionList() {
    try {
        let userId = localStorage.getItem("userId");
        const response = await axios.get(import.meta.env.VITE_URL+'/llm/history-list', {
            params: { userId: userId }
        });
        historySessions.value = response.data.historySessions;  // 更新historySessions
        console.log(response);
    } catch (error) {
        ElMessage.error({
            message: '历史记录获取失败',
            duration: 1500
        });
        console.error('请求错误:', error);
    }
}
async function getSession(sessionId) {
    try {
        const response = await axios.get(import.meta.env.VITE_URL+'/llm/getSession', {
            params: { sessionId: sessionId }
        });
        console.log(response.data)
        messages.value = response.data.messages;
        sessionName.value = response.data.session_name
        currentSessionId = response.data.session_id
        saveSessionId();
    } catch (error) {
        ElMessage.error({
            message: '历史会话获取失败',
            duration: 1500
        });
        console.error('请求错误:', error);
    }
}
async function getSessionId() {
    try {
        let userId = localStorage.getItem("userId");
        const response = await axios.get(import.meta.env.VITE_URL+'/llm/getSessionId', {
            params: { user_id: userId, session_name: sessionName.value }
        });
        currentSessionId = response.data.session_id;
        saveSessionId();
        console.log("成功获取会话id：" + currentSessionId);
    } catch (error) {
        ElMessage.error({
            message: '获取会话id失败',
            duration: 1500
        });
        console.error('请求错误:', error);
    }
}
function newSession() {
    // window.location.reload();
    currentSessionId = -1;
    saveSessionId();
    messages.value = [];
    callword.value = {
        type: '',
        definition: '',
        example: ''
    };
    params.value = {
        param1: 0,
        param2: 5,
        solution: []
    };
    files.value = [];
    if (uploadRef.value) {
        uploadRef.value.clearFiles();
    }
    sessionName.value = '未命名会话';

}

let currentSessionId = -1;
// 从 localStorage 中获取 currentSessionId
const loadSessionId = () => {
    const savedSessionId = localStorage.getItem('currentSessionId')
    if (savedSessionId !== null) {
        currentSessionId = parseInt(savedSessionId, 10)
    }
}

// 将 currentSessionId 保存到 localStorage
const saveSessionId = () => {
    localStorage.setItem('currentSessionId', currentSessionId)
}

const messages = ref([
    // { text: "帮我对以下文本进行实体抽取：陈昌旭会见贵州电网公司董事长吴国沛一行。", fromUser: true, filename: '' },
    // { text: "好的，抽取结果如下。自然人：陈昌旭 吴国沛；法人：贵州电网公司。", fromUser: false, filename: '' },
    // {text:"帮我对txt中的文本进行实体抽取.",fromUser:true,filename:'test.txt'},
    // {text:"好的，文件分析结果如下。自然人：陈昌旭 吴国沛；法人：贵州电网公司。",fromUser:false,filename:''},
    // { text: '你好呀', fromUser: true, filename: '' },
    // { text: '# Hello, **Markdown**! \nThis is a message with `code`.', fromUser: false, filename: '' }
]);
const newMessage = ref('');
const isInputDisabled = ref(false)
// 给大模型发消息，分为第一次会话的第一条、第一次会话的第n条、历史某次会话的第n条
async function sendMessage() {
    if (newMessage.value.trim()) {
        isInputDisabled.value = true;// 禁用输入框
        let userId = localStorage.getItem("userId");
        try {
            const response = await axios.post(import.meta.env.VITE_URL+'/llm/new-question', {
                old_messages: messages.value,
                new_message: newMessage.value,
                session_id: currentSessionId,
                session_name: sessionName.value,
                user_id: userId,
                settings: {
                    temperature: 0,
                    defn: callword.definition,
                    exemplar: callword.example,
                    coT: params.value.solution.includes("思维链"),
                    tf: false,
                    cws: params.value.solution.includes("分词"),
                    sv: false,
                    sc: params.value.solution.includes("多数投票"),
                    k: params.value.param2,
                    all_labels: callword.type
                },
                file: files.value.length == 0 ? "" : files.value[0].filePath,
                file_id: files.value.length == 0 ? "" : files.value[0].file_id,
                task: taskType.value
            });

            messages.value.push({ text: newMessage.value, fromUser: true, filename: files.value.length == 0 ? "" : files.value[0].filename });
            //暂定为后端反馈最新的回复和会话id
            currentSessionId = response.data.sessionId
            saveSessionId()
            messages.value.push({ text: response.data.answer, fromUser: false, filename: "" });

            newMessage.value = '';

            if (uploadRef.value) {
                uploadRef.value.clearFiles();
            }
            files.value.length = 0;
            scrollToEnd();
            isInputDisabled.value = false; 
            // ElMessage.success({
            //     message: '加载成功',
            //     duration: 1500
            // });
            // console.log(response);
        } catch (error) {
            ElMessage.error({
                message: '发送消息失败',
                duration: 1500
            });
            isInputDisabled.value = false;
            console.error('请求错误:', error);
        }
    }
}


// 提示词对话框
const callwordDialogVisible = ref(false);
const callword = reactive({
    type: [],
    definition: [],
    example: []
});
const dailog_callword = reactive({
    ...callword
})
function showCallword() {
    dailog_callword.type = [...callword.type]; // 深拷贝数组，避免直接引用同一个数组
    dailog_callword.definition = [...callword.definition];
    dailog_callword.example = [...callword.example];
    console.log(dailog_callword.type);
    console.log(dailog_callword.definition);
    callwordDialogVisible.value = true;
}
function saveCallword() {
    // 保存设置逻辑
    callword.type = [...dailog_callword.type];
  callword.definition = [...dailog_callword.definition];
  callword.example = [...dailog_callword.example];
    console.log('11111',callword,dailog_callword)
    callwordDialogVisible.value = false;
}
const testValue=ref('1')
function resetCallword() {
    console.log("chongzhi",dailog_callword)
    // 重置 callword
    callword.type.splice(0);
    callword.definition.splice(0);
    callword.example.splice(0);
    // 重置 dailog_callword
    dailog_callword.type.splice(0);
    dailog_callword.definition.splice(0);
    dailog_callword.example.splice(0);

    testValue.value='2';
    console.log(dailog_callword.type);
}


// 方法参数对话框
const paramsDialogVisible = ref(false)
// 保存的参数
const params = ref({
    param1: 0,
    param2: 5,
    solution: []
})
// 对话框显示的参数
const dailog_params = ref({
    param1: 0,
    param2: 5,
    solution: []
})
function showParams() {
    console.log(params.value);
    dailog_params.value = params.value;
    paramsDialogVisible.value = true;
    // console.log(dailog_params.value);
}
function confirmParams() {
    console.log(dailog_params.value);
    params.value = dailog_params.value;
    paramsDialogVisible.value = false;
}
function resetParams() {
    // 重置 params
    params.value = {
        param1: 0,
        param2: 5,
        solution: []
    };
    // 重置 dailog_params
    dailog_params.value = {
        param1: 0,
        param2: 5,
        solution: []
    };
}
const handleParamsChange = () => {
    // if (dailog_params.value.solution.includes("多数投票")) {
    //     if (dailog_params.value.param1 == 0) {
    //         ElMessage.warning({
    //             message: '多数投票时temperature须大于0',
    //             duration: 1500
    //         });
    //         dailog_params.value.param1 = 1;
    //     }
    // }
}

// 上传文件
const files = ref([]);
const uploadRef = ref(null);
const uploadDialogVisible = ref(false);
function showUpload() {
    uploadDialogVisible.value = true
}
const handleSuccess = (response, file, fileList) => {
    // 更新 files
    files.value = fileList.map(f => ({
        filename: f.name,
        filePath: f.response?.filePath || '',
        file_id: f.response?.file_id || ''
    }));
    ElMessage.success({
        message: '文件上传成功!',
        duration: 1500
    });
};
const handleError = (err, file, fileList) => {
    // 处理上传失败的情况
    console.error('Upload error:', err.error);
    // 可以在这里更新 UI、通知用户等
    ElMessage.error({
        message: '文件上传失败!',
        duration: 1500
    });
};
const handleRemove = async (file, fileList) => {
    try {
        // 发送删除请求到后端
        await axios.post(import.meta.env.VITE_URL+'/delete-file', {
            filePath: file.response?.filePath || '', // 假设文件路径在 file.response.filePath 中
            file_id: file.response?.file_id || ''
        });
        ElMessage.success({
            message: '文件删除成功!',
            duration: 1500
        });
        // 更新 files
        files.value = fileList.map(f => ({
            filename: f.name,
            filePath: f.response?.filePath || ''
        }));
        console.log(files.value)
    } catch (error) {
        console.error('删除文件时出错:', error);
        ElMessage.error({
            message: '文件删除失败!',
            duration: 1500
        });
    }
};
// 对话框上方文件tag
const handleFileTagClose = async (file) => {
    try {
        // 发送删除请求到后端
        await axios.post(import.meta.env.VITE_URL+'/delete-file', {
            filePath: file.filePath,
            file_id: file.file_id
        });
        ElMessage.success({
            message: '文件删除成功!',
            duration: 1500
        });
        // 更新 files
        files.value.splice(files.value.indexOf(file), 1)
        // // 这里直接删除所有的，因为限制上传一个文件
        if (uploadRef.value) {
            uploadRef.value.clearFiles();
        }
    } catch (error) {
        console.error('删除文件时出错:', error);
        ElMessage.error({
            message: '文件删除失败!',
            duration: 1500
        });
    }
}

// 会话名称下拉框
const editDialogVisible = ref(false)
const deleteDialogVisible = ref(false)
const newName = ref("")
const sessionName = ref('未命名会话');
const handleCommand = (command) => {
    if (command === 'edit') {
        editDialogVisible.value = true;
        newName.value = sessionName.value;
    } else if (command === 'delete') {
        deleteDialogVisible.value = true;
    }
}
async function delSession(sessionId) {
    try {
        let userId = localStorage.getItem("userId");
        const response = await axios.get(import.meta.env.VITE_URL+'/llm/delSession', {
            params: { sessionId: sessionId, userId: userId }
        });
        let message = response.data.message;
        ElMessage.success({
            message: message,
            duration: 1500
        });
        newSession();
        console.log(message);
    } catch (error) {
        ElMessage.error({
            message: '删除失败',
            duration: 1500
        });
        console.error('请求错误:', error);
    }
}
async function setSessionName(sessionId) {
    try {
        let userId = localStorage.getItem("userId");
        const response = await axios.get(import.meta.env.VITE_URL+'/llm/setSessionName', {
            params: {
                sessionId: sessionId,
                sessionName: newName.value,
                user_id: userId
            }
        });
        let message = response.data.message;
        ElMessage.success({
            message: message,
            duration: 1500
        });
        sessionName.value = newName.value;
        newName.value = '';
        // console.log(response);
    } catch (error) {
        ElMessage.error({
            message: '重命名失败',
            duration: 1500
        });
        console.error('请求错误:', error);
    }
}
function handleEditDialogClose() {
    newName.value = '';
    editDialogVisible.value = false;
}
async function confirmEdit() {
    if (currentSessionId > 0) {
        await setSessionName(currentSessionId);
    }
    else {
        await getSessionId();
        if (currentSessionId > 0) await setSessionName(currentSessionId);
    }
    handleEditDialogClose();
}

function handleDeleteDialogClose() {
    deleteDialogVisible.value = false;
}
async function confirmDelete() {
    if (currentSessionId > 0) {
        await delSession(currentSessionId);
    }
    else {
        ElMessage({
            message: '未创建新的会话',
            type: 'warning',
            duration: 1500
        });
    }
    handleDeleteDialogClose();
}

// 任务类型下拉框
const taskType = ref('NER')
const handleTaskType = (command) => {
    // 选中对应的下拉框的元素
    // const dropdownList = document.querySelectorAll('.el-dropdown-menu__item');
    // const dropdown3 = dropdownList[3]
    // const dropdown4 = dropdownList[4]
    if (command === 'ner'&&taskType.value != 'NER') {
        resetCallword();
        resetParams();
        taskType.value = 'NER';
    } else if (command === 'dialog'&&taskType.value != 'QA') {
        resetCallword();
        resetParams();
        taskType.value = 'QA';
    } else if (command === 're'&&taskType.value != 'RE') {
        resetCallword();
        resetParams();
        taskType.value = 'RE';
    }
}

// 滚动到页面底部,需要调整
const conversation = ref(null);
function scrollToEnd() {
    nextTick(() => {
        const endOfMessages = document.querySelector('div[id="endOfMessages"]');
        if (endOfMessages && conversation.value && conversation.value.offsetHeight > 622) {
            endOfMessages.scrollIntoView({ behavior: 'smooth' });
        }
    });
}

onMounted(() => {
    loadSessionId();
    if (currentSessionId > 0) {
        console.log(currentSessionId);
        getSession(currentSessionId);
    }
})

</script>

<style scoped>
/* 按钮通用样式 */
.icon-button {
    display: flex;
    justify-content: center;
    align-items: center;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1.5em;
    color: #5e6772;
    border-radius: 5px;
    background-color: #fff;
}

.icon-button:hover {
    background-color: #babfc4;
    color: #010101;
}

.chatbot {
    position: relative;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.sidebar {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: fixed;
    top: 35%;
    left: 270px;
    width: 60px;
    background-color: #f5f5f5;
    border-radius: 12px;
    box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
}

.sidebar .icon-button {
    height: 36px;
    width: 36px;
    margin: 5px;
}

.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    /* margin-left: 25%;
    width: 50%; */
    align-items: center;
    margin-top: 50px;
}

/* 头部 */
.header {
    display: flex;
    align-items: center;
    padding: 10px;
    width: 70%;
    flex-direction: row;
    justify-content: space-between;
    position: fixed;
    background-color: #f0f2f5;
}

.header .el-dropdown:first-child {
    width: 140px;
}

.header .custom-button {
    background-color: #f5f5f5;
    border: 0;
    color: #010101;
}

.header .custom-button:hover {
    background-color: #e5e5e5;
}

/* 下拉框 */
:deep .el-dropdown-menu__item:hover {
    background-color: #e5e5e5;
    color: #5e6772;
}

/* 对话 */
.conversation {
    display: flex;
    /* flex: 1; */
    padding: 20px;
    padding-top: 10px;
    padding-bottom: 170px;
    overflow-y: auto;
    flex-direction: column;
}

.conversation .contain {
    display: flex;
    overflow-y: auto;
    flex-direction: column;
    width: 600px;
}

.message {
    margin-bottom: 10px;
    padding: 10px;
    padding-bottom: 0;
    border-radius: 12px;
    background-color: #fff;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.message.from-user {
    background-color: #d1e7ff;
    align-self: flex-end;
    max-width: 60%;
}

.file-message {
    align-self: flex-end;
    max-width: 60%;
    margin-top: -5px;
    padding: 3px 10px;
    font-size: 12px;
    border-radius: 5px;
    background-color: #dedfe9;
}

.filelist-tag {
    display: flex;
    position: fixed;
    bottom: 143px;
    width: 688px;
}

.input-box {
    display: flex;
    width: 688px;
    position: fixed;
    bottom: 0;
    align-items: center;
    padding-bottom: 15px;
    background-color: #f0f2f5;
}

:deep .input-text .el-textarea__inner {
    border-radius: 12px;
    resize: none;
    padding: 15px 15px 45px 15px;
}

/* 输入框按钮样式 */
.input-box .upfile-button {
    right: 40px;
}

.input-box .icon-button:last-of-type {
    right: 5px;
}

.input-box .icon-button {
    height: 30px;
    width: 30px;
    bottom: 20px;
    position: absolute;
    z-index: 2;
}

/* 对话框样式 */
:deep .el-form-item__label {
    font-size: 16px;
}

:deep .el-dialog__headerbtn {
    outline: none;
}

.setting-input {
    width: 260px;
}

/* 历史会话 */
.history-menu {
    max-height: 350px;
    /* Set the fixed height */
    overflow-y: auto;
    /* Enable vertical scrolling */
}

.el-menu-item {
    border-radius: 8px;
    display: flex;
    justify-content: space-between;
}

.el-menu-item:hover {
    color: #003f88 !important;
    /* 鼠标悬停时蓝色 */
    background-color: #ececec;
}

.el-menu-item.is-active {
    background-color: #ececec !important;
    /* 鼠标点击后的背景颜色 */
    color: #003f88 !important;
    /* 鼠标点击后的文字颜色 */
}

/* 提示词对话框 */
.tags {
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
}

.tags .el-tag {
    margin-right: 5px;
    margin-bottom: 5px;
}
</style>
