<template>
  <el-container>
    <el-header>
      <el-page-header title="机器学习实践平台" />
    </el-header>
    
    <el-main>
      <el-row :gutter="20">
        <!-- 左侧配置面板 -->
        <el-col :span="8">
          <el-card class="config-panel">
            <template #header>
              <div class="card-header">
                <span>配置选项</span>
                <el-tag type="info">已连接: {{connectUrl}}:{{connectPort}}</el-tag>
              </div>
            </template>
            
            <div v-for="(value, key) in configDict" :key="key">
              <h3>{{ key }}</h3>
              <el-radio-group v-model="configValue[key]" class="radio-group">
                <el-radio-button v-for="name in value" :key="name" :label="name">
                  {{ name }}
                </el-radio-button>
              </el-radio-group>
            </div>
            
            <el-button :icon="CaretRight" class="run-button" type="primary" @click="clickButton">
              运行
            </el-button>
          </el-card>
        </el-col>

        <!-- 右侧结果展��� -->
        <el-col :span="16">
          <el-card class="result-panel">
            <template #header>
              <div class="card-header">
                <span>运行结果</span>
                <el-button :icon="Delete" type="primary" @click="clearOutput">
                  清空
                </el-button>
              </div>
            </template>
            
            <div class="result-content">
              <el-scrollbar height="600px">
                <div v-for="(content, idx) in runningOutput" :key="idx" class="result-item">
                  <el-alert
                    v-if="content.type === 'string'"
                    :closable="false"
                    :title="content.content"
                  />
                  <el-image
                    v-if="content.type === 'image'"
                    :preview-src-list="['data:image/jpeg;base64,'+content.content]"
                    :src="'data:image/jpeg;base64,'+content.content"
                    fit="contain"
                  />
                </div>
              </el-scrollbar>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </el-main>
  </el-container>
</template>

<script lang="ts" setup>
import { ElMessage } from 'element-plus'
import { ref } from 'vue'

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

<style scoped>
.config-panel {
  height: 100%;
}

.result-panel {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.radio-group {
  margin: 10px 0;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.run-button {
  width: 100%;
  margin-top: 20px;
}

.result-content {
  min-height: 600px;
}

.result-item {
  margin: 10px 0;
}
</style>
