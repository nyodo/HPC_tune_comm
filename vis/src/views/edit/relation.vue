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
                <el-input v-model="search.query" placeholder="请输入用户名、文本或结果内容" />
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
        <el-table-column prop="username" label="用户名" width="100"/>
        <el-table-column prop="text" label="文本" width="500"/>
        <el-table-column prop="relation" label="结果" width="500"/>
        <el-table-column prop="created_time" label="创建时间" width="180" />

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
        <el-form-item label="用户名:">
          {{ form.username }}
        </el-form-item>
        <el-form-item label="文本:">
          {{ form.text }}
        </el-form-item>
        <el-form-item label="结果:">
          <el-input v-model="form.relation" type="textarea" :rows="4" placeholder="请输入内容"></el-input>
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

<script>
// 引入 utils.js 中的所有函数
import sDrawer from "@/components/s-drawer/s-drawer.vue"
export default {
  components: {
    sDrawer,
  },
  watch: {
    visible(value) {
      if (!value) {
        this.form = {}
      }
    }
  },
  data() {
    return {
      form: {
        id: 0,
        user_id: 0,
        text: '',
        relation: ''
      },
      visible: false,
      tableData: [],
      totalNum: 100,
      search: {
        pageNum: 1,
        pageSize: 10,
        query: "",
      },
      monthList: [
        { id: '1', name: '一月' },
        { id: '2', name: '二月' },
        { id: '3', name: '三月' },
        { id: '4', name: '四月' },
        { id: '5', name: '五月' },
        { id: '6', name: '六月' },
        { id: '7', name: '七月' },
        { id: '8', name: '八月' },
        { id: '9', name: '九月' },
        { id: '10', name: '十月' },
        { id: '11', name: '十一月' },
        { id: '12', name: '十二月' }
      ]
    };
  },
  computed: {

  },
  created() {
    // this.search.token = localStorage.getItem("token");
    this.getList();
  },

  methods: {
    formatIsAdmin(value) {
      return value ? '是' : '否';
    },
    async getList() {
      const res = await this.$request.get(
        import.meta.env.VITE_URL+"/relations/",
        { params: this.search }
      );

      if (res.data.code === 200) {
        // this.tableData = res.data.data.sort((a, b)      
        console.log(res.data)
        this.tableData = res.data.data;
        this.totalNum = res.data.zs
      }
    },
    // 每页条数改变时触发 选择一页显示多少行
    handleSizeChange(val) {
      console.log(`每页 ${val} 条`);
      this.search.pageSize = val;
      this.getList();
    },
    // 当前页改变时触发 跳转其他页
    handleCurrentChange(val) {
      console.log(`当前页: ${val}`);
      this.search.pageNum = val;
      this.getList();
    },
    resetSearch() {
      let search = {
        pageNum: this.search.pageNum,
        pageSize: this.search.pageSize,
      };
      this.search = search;
      this.getList();
    },
    editItem(row) {
      console.log(this.$deepClone(row))
      this.form = this.$deepClone(row)
      this.visible = true
    },
    async saveData() {
      console.log(this.form)
      if (this.form.id) {
        const res = await this.$request.put(import.meta.env.VITE_URL+'/relations/' + this.form.id + '/', this.form)
        if (res.data.code === 200) {
          this.$message.success(res.data.data)
          this.getList()
          this.visible = false
        }
      } else {
        const res = await this.$request.post(import.meta.env.VITE_URL+'/relations/', this.form)
        if (res.data.code === 200) {
          this.$message.success(res.data.data)
          this.getList()
          this.visible = false
        }
      }
    },
    async deleteItem(id) {
      const res = await this.$request.delete(import.meta.env.VITE_URL+"/relations/" + id + '/')
      if (res.data.code === 200) {
        this.$message.success(res.data.data)
        this.getList()
      }
    },
  },
};
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