<template>
  <div class="edit-container">
    <!-- <el-button type="primary" @click="visible = true" disabled>添加记录</el-button> -->
    <el-card shadow="never" style="margin-top: 10px;">

      <!-- 查询区域 -->
      <div style="margin: 10px 10px 0 10px;">
        <el-form :model="search">
          <el-row :gutter="10">
            <el-col :span="6">
              <el-form-item label="搜索:">
                <el-input v-model="search.query" placeholder="用户名、会话名、问答或任务类型" />
              </el-form-item>
            </el-col>
            <el-col :span="6">
              <div style="text-align: right;">
                  <el-button type="primary" @click="getList()">查询</el-button>
                  <el-button @click="resetSearch()">重置</el-button>
              </div>
            </el-col>
          </el-row>
        </el-form>
      </div>

      <el-divider />

      <!-- 表格区域 -->
      <el-table :data="tableData" stripe>
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column prop="session_name" label="会话名" width="100"/>
        <el-table-column prop="username" label="用户名" width="100"/>
        <el-table-column prop="question" label="提问" width="500"/>
        <el-table-column prop="answer" label="回答" width="500"/>
        <el-table-column prop="filename" label="文件名" width="150"/>
        <el-table-column prop="task" label="任务类型" width="80"/>
        <el-table-column prop="created_time" label="创建时间" width="160" />

        <el-table-column label="操作" width="120" fixed="right">
          <template #default="scope">
            <el-button link type="primary" size="small" @click="editItem(scope.row)" >编辑</el-button>
            <el-popconfirm title="确定要删除吗?" @confirm="deleteItem(scope.row.id)">
              <template #reference>
                <el-button link type="primary" size="small">删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>

      </el-table>
      <!-- 分页 -->
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="sizes, total, prev, pager, next" :total="totalNum" :currentPage="search.pageNum"
        :pageSize="search.pageSize">
      </el-pagination>
    </el-card>
    <!-- 新增或编辑的抽屉 -->
    <sDrawer v-model="visible" :title="form.id ? '编辑记录' : '添加记录'" size="35%" :close-on-click-modal="false">

      <el-form :model="form" label-width="100px" class="drawer-form">
        <el-form-item label="ID:">
          {{ form.id }}
        </el-form-item>
        <el-form-item label="会话名:">
          {{ form.session_name }}
        </el-form-item>
        <el-form-item label="用户名:">
          {{ form.username }}
        </el-form-item>
        <el-form-item label="提问:">
          {{ form.question }}
        </el-form-item>
        <el-form-item label="回答:">
          <el-input v-model="form.answer" type="textarea" :rows="4" placeholder="请输入内容"></el-input>
        </el-form-item>
        <el-form-item label="文件名:">
          {{ form.filename }}
        </el-form-item>
        <el-form-item label="任务类型:">
          {{ form.task }}
        </el-form-item>
        <el-form-item label="创建时间:">
          {{ form.created_time }}
        </el-form-item>
      </el-form>


      <template #footer>
        <el-button @click="visible = false">取消</el-button>
        <el-button type="primary" @click="saveData()">确定</el-button>
      </template>
    </sDrawer>
  </div>

  
</template>

<script setup>
import { ElMessage } from 'element-plus';
import { ref, watch, onMounted } from 'vue';
import sDrawer from '@/components/s-drawer/s-drawer.vue';

import { getCurrentInstance } from 'vue';
const { proxy } = getCurrentInstance();


const form = ref({
  id: 0,
  session_id:0,
  user_id: 0,
  username:'',
  question:'',
  answer:'',
  filename:'',
  task:'',
  created_time: ''
});

const visible = ref(false);
const tableData = ref([]);
const totalNum = ref(100);
const search = ref({
  pageNum: 1,
  pageSize: 10,
  query: '',
});

const formatIsAdmin = (value) => value ? '是' : '否';

const getList = async () => {
  try {
    const res = await proxy.$request.get(import.meta.env.VITE_URL+'/qas/', { params: search.value });
    if (res.data.code === 200) {
      // console.log(res.data);
      for (const qa_data of res.data.data){
        if (qa_data.filename == null){
          qa_data.filename = '无'
        }
      }
      tableData.value = res.data.data;
      totalNum.value = res.data.zs;
    }
  } catch (error) {
    console.error(error);
  }
};

const handleSizeChange = (val) => {
  // console.log(`每页 ${val} 条`);
  search.value.pageSize = val;
  getList();
};

const handleCurrentChange = (val) => {
  // console.log(`当前页: ${val}`);
  search.value.pageNum = val;
  getList();
};

const resetSearch = () => {
  search.value = {
    pageNum: search.value.pageNum,
    pageSize: search.value.pageSize,
  };
  getList();
};

const editItem = (row) => {
  form.value = { ...row }; // Using shallow copy
  visible.value = true;
};

const saveData = async () => {
  try {
    if (form.value.id) {
      const res = await proxy.$request.put(import.meta.env.VITE_URL+'/qas/' + form.value.id + '/', form.value);
      if (res.data.code === 200) {
        ElMessage.success({
          message:res.data.data,
          duration:1500
        });
        getList();
        visible.value = false;
      }
    } else {
      const res = await proxy.$request.post(import.meta.env.VITE_URL+'/qas/', form.value);
      if (res.data.code === 200) {
        ElMessage.success({
          message:res.data.data,
          duration:1500
        });
        getList();
        visible.value = false;
      }
    }
  } catch (error) {
    console.error(error);
  }
};

const deleteItem = async (id) => {
  try {
    const res = await proxy.$request.delete(import.meta.env.VITE_URL+'/qas/' + id + '/');
    if (res.data.code === 200) {
      ElMessage.success({
          message:res.data.data,
          duration:1500
        });
      getList();
    }
  } catch (error) {
    console.error(error);
  }
};

watch(visible, (value) => {
  if (!value) {
    form.value = {
      id: 0,
      user_id: 0,
      text: '',
      relation: ''
    };
  }
});

onMounted(() => {
  getList();
});
</script>

<style scoped>
.edit-container {
  padding: 20px 20px 0 20px;
}

.el-pagination {
  margin-top: 10px;
}

.el-row {
  margin-bottom: 20px;
}

.el-row:last-child {
  margin-bottom: 0px;
}

.drawer-form .el-form-item .el-input,
.drawer-form .el-form-item .el-select {
  width: 250px;
}

/* el-divider 修改高度&虚线效果 */
.el-divider--horizontal{
    margin: 2px 0;
    background: 0 0;
    border-top: 1px dashed #e8eaec;
} 
</style>