<template>
  <el-container class="main-container">
    <!-- 顶部导航栏 -->
    <el-header class="header">
      <div class="header-content">
        <div class="logo-title">
          <img alt="Logo" class="logo" src="/logo.ico">
          <h1>机器学习实践平台</h1>
        </div>
        <div class="header-right">
          <el-tooltip content="开发者信息" placement="bottom">
            <el-button
              class="developer-btn"
              link
              type="info"
              @click="showDeveloperInfo = true"
            >
              <el-icon><User /></el-icon>
            </el-button>
          </el-tooltip>
          <el-tag class="connection-status" effect="dark" type="success">
            <el-icon><Connection /></el-icon>
            已连接: {{connectUrl}}:{{connectPort}}
          </el-tag>
        </div>
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
                  <el-icon class="header-icon" @click="showSettings = true"><Setting /></el-icon>
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
          <el-row :gutter="20">
            <!-- 文本输出面板 -->
            <el-col :span="14">
              <el-card class="result-panel" shadow="hover">
                <template #header>
                  <div class="panel-header">
                    <div class="title-section">
                      <el-icon><Document /></el-icon>
                      <span class="title">评价结果</span>
                    </div>
                    <div class="controls">
                      <el-dropdown style="margin-right: 10px" @command="handleExport">
                        <el-button size="small" type="primary">
                          导出结果
                          <el-icon class="el-icon--right"><Download /></el-icon>
                        </el-button>
                        <template #dropdown>
                          <el-dropdown-menu>
                            <el-dropdown-item command="text">导出文本</el-dropdown-item>
                            <el-dropdown-item command="image">导出图片</el-dropdown-item>
                            <el-dropdown-item command="all">导出全部</el-dropdown-item>
                          </el-dropdown-menu>
                        </template>
                      </el-dropdown>
                      <el-button 
                        :icon="Delete"
                        size="small"
                        type="danger"
                        @click="clearTextOutput">
                        清空文本
                      </el-button>
                    </div>
                  </div>
                </template>
                
                <div class="result-content">
                  <el-scrollbar height="600px">
                    <transition-group name="fade">
                      <div v-for="(content, idx) in textOutput" 
                           :key="idx" 
                           class="result-item">
                        <el-alert
                          :closable="false"
                          :title="content"
                          class="text-output"
                        />
                      </div>
                    </transition-group>
                  </el-scrollbar>
                </div>
              </el-card>
            </el-col>

            <!-- 图片输出面板 -->
            <el-col :span="10">
              <el-card class="result-panel" shadow="hover">
                <template #header>
                  <div class="panel-header">
                    <div class="title-section">
                      <el-icon><Picture /></el-icon>
                      <span class="title">可视化输出</span>
                    </div>
                    <div class="controls">
                      <el-button 
                        :icon="Delete"
                        size="small"
                        type="danger"
                        @click="clearImageOutput">
                        清空图片
                      </el-button>
                    </div>
                  </div>
                </template>
                
                <div class="result-content">
                  <el-scrollbar height="600px">
                    <transition-group name="fade">
                      <div v-for="(content, idx) in imageOutput" 
                           :key="idx" 
                           class="result-item">
                        <el-image
                          :preview-src-list="['data:image/jpeg;base64,'+content]"
                          :src="'data:image/jpeg;base64,'+content"
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
        </el-col>
      </el-row>
    </el-main>
  </el-container>

  <!-- 添加设置对话框 -->
  <el-dialog
    v-model="showSettings"
    title="系统设置"
    width="500px"
    :close-on-click-modal="false"
  >
    <el-tabs v-model="activeSettingTab">
      <!-- 显示设置选项卡 -->
      <el-tab-pane label="显示设置" name="display">
        <el-form label-width="120px">
          <el-form-item label="图像输出">
            <el-switch
              v-model="displaySettings.showImages"
              active-text="启用"
              inactive-text="禁用"
            />
          </el-form-item>
          <el-form-item label="图像大小限制">
            <el-input-number 
              v-model="displaySettings.maxImageHeight" 
              :disabled="!displaySettings.showImages"
              :max="800"
              :min="200"
              :step="50"
            />
            <span class="unit-text">px</span>
          </el-form-item>
        </el-form>
      </el-tab-pane>
      
      <!-- 连接设置选项卡 -->
      <el-tab-pane label="连接设置" name="connection">
        <el-form :model="connectionForm" label-width="100px">
          <el-form-item label="服务器地址">
            <el-input v-model="connectionForm.url" placeholder="请输入服务器地址" />
          </el-form-item>
          <el-form-item label="端口">
            <el-input v-model="connectionForm.port" placeholder="请输入端口号" />
          </el-form-item>
        </el-form>
      </el-tab-pane>
      
      <!-- 按钮设置选项卡 -->
      <el-tab-pane label="按钮设置" name="buttons">
        <div class="button-settings">
          <div v-for="(section, key) in configDict" :key="key" class="button-section">
            <div class="section-header">
              <h3>{{ key }}</h3>
              <el-popover
                v-if="deletedButtons[key]?.length"
                placement="bottom"
                :width="200"
                trigger="click"
              >
                <template #reference>
                  <el-button 
                    size="small"
                    type="success"
                  >
                    恢复按钮
                  </el-button>
                </template>
                <div class="deleted-buttons-list">
                  <div v-for="(btn, idx) in deletedButtons[key]" :key="idx" class="deleted-button-item">
                    <span>{{ btn }}</span>
                    <el-button 
                      size="small"
                      type="primary"
                      @click="restoreButton(key, idx)"
                    >
                      恢复
                    </el-button>
                  </div>
                </div>
              </el-popover>
            </div>
            <el-table :data="section" style="width: 100%">
              <el-table-column label="按钮名称" prop="name">
                <template #default="{ row, $index }">
                  <el-input v-model="section[$index]" placeholder="输入按钮名称" />
                </template>
              </el-table-column>
              <el-table-column label="操作" width="120">
                <template #default="{ row, $index }">
                  <el-button 
                    size="small"
                    type="danger"
                    @click="removeButton(key, $index, row)"
                  >
                    删除
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>
    
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="showSettings = false">取消</el-button>
        <el-button type="primary" @click="handleSaveSettings">
          保存设置
        </el-button>
      </span>
    </template>
  </el-dialog>

  <!-- 添加开发者信息对话框 -->
  <el-dialog
    v-model="showDeveloperInfo"
    title="开发者信息"
    width="400px"
  >
    <div class="developer-info">
      <h3>YOKI LEE</h3>
      <p>CUC - 202212033024</p>
      <p>邮箱：642814925@qq.com</p>
      <p>Gitee:https://gitee.com/yokilee</p>
    </div>
  </el-dialog>
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

.logo-title h1 {
  margin: 0;
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

.header-icon {
  cursor: pointer;
  font-size: 20px;
  color: #409EFF;
  transition: color 0.3s;
}

.header-icon:hover {
  color: #66b1ff;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.button-settings {
  max-height: 400px;
  overflow-y: auto;
}

.button-section {
  margin-bottom: 20px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.section-header h3 {
  margin: 0;
  color: #606266;
}

.deleted-buttons-list {
  max-height: 200px;
  overflow-y: auto;
}

.deleted-button-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #EBEEF5;
}

.deleted-button-item:last-child {
  border-bottom: none;
}

.deleted-button-item span {
  color: #606266;
  margin-right: 10px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 15px;
}

.developer-btn {
  font-size: 18px;
  color: #909399;
}

.developer-btn:hover {
  color: #409EFF;
}

.developer-info {
  padding: 10px;
}

.developer-info h3 {
  color: #303133;
  margin: 10px 0;
  font-size: 16px;
}

.developer-info p {
  color: #606266;
  margin: 5px 0;
  font-size: 14px;
}

.el-divider {
  margin: 15px 0;
}

.result-panel {
  margin-bottom: 20px;
}

.image-output {
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  margin: 10px 0;
  cursor: pointer;
}

.text-output {
  margin: 10px 0;
}

.result-panel {
  height: calc(100vh - 140px);
  overflow: hidden;
}

.result-content {
  height: calc(100% - 60px);
  padding: 10px;
}

.el-scrollbar {
  height: 100% !important;
}

/* 添加图片预览相关样式 */
:deep(.el-image-viewer__wrapper) {
  z-index: 2100;
}

:deep(.el-image-viewer__img) {
  max-width: 90%;
  max-height: 90%;
}

.unit-text {
  margin-left: 10px;
  color: #909399;
}

/* 修改图片输出样式，使用动态高度 */
.image-output {
  width: 100%;
  max-height: v-bind('displaySettings.showImages ? displaySettings.maxImageHeight + "px" : "0px"');
  object-fit: contain;
  margin: v-bind('displaySettings.showImages ? "10px 0" : "0"');
  cursor: pointer;
  transition: all 0.3s ease;
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
  DataAnalysis,
  User,
  Download,
  Document,
  Picture
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
    
    if (received_msg.status === 200) {
      const received_msg_data = received_msg.data
      if (received_msg.type === 'overview') {
        configDict.value = received_msg.data
        
        const tmpDict: DlFrameConfigInterface = {}
        for (let i in configDict.value) {
          tmpDict[i] = ''
        }
        configValue.value = tmpDict
        textOutput.value = []
        imageOutput.value = []
      }
      else if (received_msg.type === 'print') {
        textOutput.value.push(received_msg_data.content)
      }
      else if (received_msg.type === 'imshow') {
        if (displaySettings.value.showImages) {
          imageOutput.value.push(received_msg_data.content)
        }
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
    // 清空之前的输出
    textOutput.value = []
    imageOutput.value = []
    
    ws?.send(JSON.stringify({
      'type': 'run', 
      'params': configValue.value
    }))
  } finally {
    isRunning.value = false
  }
}

const isRunning = ref(false)

const autoReconnect = () => {
  if (!isConnectedToServer.value) {
    setTimeout(() => {
      connect()
    }, 5000)
  }
}

// 添加新的响应式变量
const showSettings = ref(false)
const connectionForm = ref({
  url: connectUrl.value,
  port: connectPort.value
})

// 添加新的响应式变量
const activeSettingTab = ref('connection')

// 添加按钮管理方法
const addButton = (section: string) => {
  if (Array.isArray(configDict.value[section])) {
    configDict.value[section].push('')
  }
}

// 添加已删除按钮存储
interface DeletedButtonsInterface {
  [key: string]: string[];
}

const deletedButtons = ref<DeletedButtonsInterface>({})

// 修改删除按钮的方法
const removeButton = (section: string, index: number, value: string) => {
  if (Array.isArray(configDict.value[section])) {
    // 保存删除的按钮
    if (!deletedButtons.value[section]) {
      deletedButtons.value[section] = []
    }
    deletedButtons.value[section].push(configDict.value[section][index])
    
    // 从当前列表中删除
    configDict.value[section].splice(index, 1)
  }
}

// 添加恢复按钮的方法
const restoreButton = (section: string, deletedIndex: number) => {
  if (deletedButtons.value[section] && deletedButtons.value[section][deletedIndex]) {
    // 恢复按钮到原列表
    configDict.value[section].push(deletedButtons.value[section][deletedIndex])
    
    // 从已删除列表中移除
    deletedButtons.value[section].splice(deletedIndex, 1)
    
    // 如果该分类下没有已删除按钮了，删除该分类
    if (deletedButtons.value[section].length === 0) {
      delete deletedButtons.value[section]
    }
    
    ElMessage.success('按钮已恢复')
  }
}

// 修改保存设置的方法
const handleSaveSettings = () => {
  // 保存连接设置
  connectUrl.value = connectionForm.value.url
  connectPort.value = connectionForm.value.port
  
  // 保存按钮设置到服务器
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      'type': 'update_config',
      'params': configDict.value
    }))
  }
  
  showSettings.value = false
  
  // 如果禁用图像输出，清空现有图像
  if (!displaySettings.value.showImages) {
    imageOutput.value = []
  }
  
  // 如果连接设置发生改变，重新连接
  if (connectUrl.value !== connectionForm.value.url || 
      connectPort.value !== connectionForm.value.port) {
    if (ws) {
      ws.close()
    }
    connect()
  }
}

// 添加 WebSocket 消息处理
const handleWebSocketMessage = (received_msg: any) => {
  if (received_msg.status === 200) {
    if (received_msg.type === 'update_config') {
      ElMessage.success('配置更新成功')
    }
    // ... 其他现有的消息处理 ...
  }
}

// 添加开发者信息控制变量
const showDeveloperInfo = ref(false)

// 添加导出功能
const handleExport = (type: string) => {
  switch (type) {
    case 'text':
      if (textOutput.value.length === 0) {
        ElMessage.warning('没有可导出的文本')
        return
      }
      exportText()
      break
    case 'image':
      if (imageOutput.value.length === 0) {
        ElMessage.warning('没有可导出的图片')
        return
      }
      exportImages()
      break
    case 'all':
      if (textOutput.value.length === 0 && imageOutput.value.length === 0) {
        ElMessage.warning('没有可导出的内容')
        return
      }
      exportAll()
      break
  }
}

// 导出文本结果
const exportText = () => {
  if (textOutput.value.length === 0) {
    ElMessage.warning('没有可导出的文本')
    return
  }

  const textContent = textOutput.value.join('\n')
  const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `实验结果_${new Date().toLocaleString()}.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// 导出图片结果
const exportImages = () => {
  if (imageOutput.value.length === 0) {
    ElMessage.warning('没有可导出的图片')
    return
  }

  imageOutput.value.forEach((content, index) => {
    const link = document.createElement('a')
    link.href = `data:image/jpeg;base64,${content}`
    link.download = `实验图片_${index + 1}.jpg`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  })
}

// 导出所有结果
const exportAll = () => {
  exportText()
  exportImages()
}

// 分离文本和图片输出
const textOutput = ref<string[]>([])
const imageOutput = ref<string[]>([])

// 分别清空文本和图片输出
const clearTextOutput = () => {
  textOutput.value = []
}

const clearImageOutput = () => {
  imageOutput.value = []
}

// 添加显示设置
const displaySettings = ref({
  showImages: true,
  maxImageHeight: 400
})
</script>
