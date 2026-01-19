<template>
  <div class="taginput">
    <el-input
      v-model="input"
      placeholder=""
    >
      <template #append>
        <el-button @click="addTag">添加</el-button>
      </template>
    </el-input>
    <div class="tags">
      <el-tag
        v-for="(tag, index) in tags"
        :key="index"
        closable
        @close="removeTag(index)"
      >
        {{ tag }}
      </el-tag>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  modelValue: {
    type: Array,
    required: true
  },
  test:{
    type:String
  }
})

const emit = defineEmits(['update:modelValue'])

const input = ref('')
const tags = ref([...props.modelValue])
watch(tags.value, (newTags) => {
  emit('update:modelValue', newTags)
  console.log(newTags,"newTags");
})
watch(()=>props.test,(newValue)=>{
  console.log('test:',newValue);
})
watch(()=>props.modelValue,(newValue)=>{
  console.log('newValue',newValue);
})
const addTag = () => {
  if (input.value.trim() && !tags.value.includes(input.value.trim())) {
    tags.value.push(input.value.trim())
    input.value = ''
  }
}

const removeTag = (index) => {
  tags.value.splice(index, 1)
}

</script>

<style scoped>
.taginput {
  width: 100%;
  min-width: 100px;
  max-width: 274px;
  word-wrap: break-word; /* 自动换行 */
}

.tags {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
}

.tags .el-tag {
  margin-right: 5px;
  margin-bottom: 5px;
  max-width: 100%; /* 限制标签最大宽度 */
  white-space: normal; /* 使标签中的内容能够换行 */
  line-height: normal;
  height:auto;
  padding: 4px 8px; /* 可根据需要调整内边距，确保内容有足够的空间 */
}
</style>
