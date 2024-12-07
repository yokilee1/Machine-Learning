<template>
  <el-container class="main-container">
    <!-- 顶部导航栏 -->
    <el-header class="header">
      <div class="header-content">
        <div class="logo-title">
          <img alt="Logo" class="logo" src="/logo.ico">
          <h1>机器学习实践平台</h1>
        </div>
        <el-tag class="connection-status" effect="dark" type="success">
          <el-icon><Connection /></el-icon>
          已连接: {{connectUrl}}:{{connectPort}}
        </el-tag>
      </div>
    </el-header>
    
    <el-main class="main-content">
      <el-row :gutter="20">
        <!-- 左侧配置面板 -->
        <el-col :span="8">
          <el-card class="config-panel" shadow="hover">
            <template #header>
              <div class="panel-header">
                <span class="title">实验配置</span>
                <el-tooltip content="配置实验参数" placement="top">
                  <el-icon class="header-icon"><Setting /></el-icon>
                </el-tooltip>
              </div>
            </template>
            
            <div class="config-sections">
              <div v-for="(value, key) in configDict" :key="key" class="config-section">
                <h3 class="section-title">
                  <el-icon><Menu /></el-icon>
                  {{ key }}
                </h3>
                <el-radio-group v-model="configValue[key]" class="radio-group">
                  <el-radio-button 
                    v-for="name in value" 
                    :key="name" 
                    :label="name"
                    class="radio-button">
                    {{ name }}
                  </el-radio-button>
                </el-radio-group>
              </div>
            </div>
            
            <el-button 
              :icon="CaretRight"
              :loading="isRunning"
              class="run-button"
              type="primary"
              @click="clickButton">
              {{ isRunning ? '运行中...' : '开始实验' }}
            </el-button>
          </el-card>
        </el-col>

        <!-- 右侧结果展示 -->
        <el-col :span="16">
          <el-card class="result-panel" shadow="hover">
            <template #header>
              <div class="panel-header">
                <div class="title-section">
                  <el-icon><DataAnalysis /></el-icon>
                  <span class="title">实验结果</span>
                </div>
                <div class="controls">
                  <el-button 
                    :icon="Delete"
                    size="small"
                    type="danger"
                    @click="clearOutput">
                    清空结果
                  </el-button>
                </div>
              </div>
            </template>
            
            <div class="result-content">
              <el-scrollbar height="600px">
                <transition-group name="fade">
                  <div v-for="(content, idx) in runningOutput" 
                       :key="idx" 
                       class="result-item">
                    <el-alert
                      v-if="content.type === 'string'"
                      :closable="false"
                      :title="content.content"
                      class="text-output"
                    />
                    <el-image
                      v-if="content.type === 'image'"
                      :preview-src-list="['data:image/jpeg;base64,'+content.content]"
                      :src="'data:image/jpeg;base64,'+content.content"
                      class="image-output"
                      fit="contain"
                    />
                  </div>
                </transition-group>
              </el-scrollbar>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<style scoped>
.main-container {
  min-height: 100vh;
  background-color: #f5f7fa;
}

.header {
  background-color: #fff;
  box-shadow: 0 2px 4px rgba(0,0,0,.1);
  padding: 0 20px;
}

.header-content {
  height: 60px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo-title {
  display: flex;
  align-items: center;
  gap: 15px;
}

.logo {
  height: 40px;
  width: 40px;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 5px;
}

.main-content {
  padding: 20px;
}

.config-panel, .result-panel {
  height: calc(100vh - 140px);
  overflow: hidden;
  transition: all 0.3s ease;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 10px;
}

.title-section {
  display: flex;
  align-items: center;
  gap: 8px;
}

.title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.config-sections {
  padding: 10px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #606266;
  margin-bottom: 15px;
}

.radio-group {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 20px;
}

.radio-button {
  margin-bottom: 10px;
}

.run-button {
  width: 100%;
  height: 40px;
  margin-top: 20px;
  font-size: 16px;
}

.result-content {
  padding: 10px;
}

.result-item {
  margin: 10px 0;
  transition: all 0.3s ease;
}

.text-output {
  border-radius: 4px;
}

.image-output {
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0,0,0,0.1);
}

/* 添加过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .el-row {
    flex-direction: column;
  }
  
  .el-col {
    width: 100% !important;
    margin-bottom: 20px;
  }
}
</style>

<script lang="ts" setup>
import { ref } from 'vue'
import { 
  CaretRight, 
  Delete, 
  Setting, 
  Menu, 
  Connection,
  DataAnalysis 
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const connectUrl = ref('localhost')
const connectPort = ref('8765')
const showConnectInfoWindow = ref(false)

interface DlFrameConfigInterface {
  [key: string]: string;
}
interface DlFrameInspectionInterface {
  [key: string]: Array<string>;
}

let ws: WebSocket | null = null
const connect = () => {
  ws = new WebSocket('ws://' + connectUrl.value + ':' + connectPort.value)
  ws.onopen = () => {
    isConnectedToServer.value = true
    showConnectInfoWindow.value = false
    ElMessage({
      message: '连接成功',
      type: 'success',
    })
    ws?.send(JSON.stringify({
      'type': 'overview', 
      'params': {}
    }))
  }
  ws.onmessage = (evt) => {
    var received_msg = JSON.parse(evt.data);
    // console.log(received_msg);
    
    if (received_msg.status === 200) {
      const received_msg_data = received_msg.data
      if (received_msg.type === 'overview') {
        // console.log(received_msg.data)
        configDict.value = received_msg.data
        
        const tmpDict: DlFrameConfigInterface = {}
        for (let i in configDict.value) {
          tmpDict[i] = ''
        }
        configValue.value = tmpDict
        runningOutput.value = []
      }

      else if (received_msg.type === 'print') {
        runningOutput.value.push({
          'type': 'string', 
          'content': received_msg_data.content
        })
      }

      else if (received_msg.type === 'imshow') {
        runningOutput.value.push({
          'type': 'image', 
          'content': received_msg_data.content
        })
      }
    } else {
      console.error(received_msg.data);
    }
  }
  ws.onclose = () => {
    isConnectedToServer.value = false
    showConnectInfoWindow.value = true
    ElMessage.error('连接已断开')
  }
  ws.onerror = () => {
    ElMessage.error('连接失败(地址错误 / 协议错误 / 服务器错误)')
    showConnectInfoWindow.value = true
  } 
}

connect()

const configDict = ref<DlFrameInspectionInterface>({})
const configValue = ref<DlFrameConfigInterface>({})

const isConnectedToServer = ref(false)
const onClickConnect = () => {
  connect()
}

const clickButton = async () => {
  if (isRunning.value) {
    ElMessage.warning('任务正在运行中...')
    return
  }
  
  try {
    isRunning.value = true
    if (!isConnectedToServer.value) {
      ElMessage.error('与 server 的连接已断开。请重启 python 服务后刷新页面')
      return
    }
    for (let k in configDict.value) {
      if (configDict.value[k].length === 0) {
        ElMessage.error('没有 ' + k)
        return
      }
    }
    for (let k in configValue.value) {
      if (configValue.value[k] == '') {
        ElMessage.error('您的选项不完整')
        return
      }
    }
    runningOutput.value = []
    ws?.send(JSON.stringify({
      'type': 'run', 
      'params': configValue.value
    }))
  } finally {
    isRunning.value = false
  }
}

interface RunningOutputInterface {
  [key: string]: string;
}
const runningOutput = ref<Array<RunningOutputInterface>>([])

const clearOutput = () => {
  runningOutput.value = []
}

const isRunning = ref(false)

const autoReconnect = () => {
  if (!isConnectedToServer.value) {
    setTimeout(() => {
      connect()
    }, 5000)
  }
}
</script>
