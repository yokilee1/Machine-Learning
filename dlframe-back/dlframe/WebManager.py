import queue
import threading
import asyncio
import json
import websockets
import traceback

from dlframe.CalculationNodeManager import CalculationNodeManager
from dlframe.Logger import Logger

class SendSocket:
    def __init__(self, socket) -> None:
        self.sendBuffer = queue.Queue()
        self.socket = socket
        self.sendThread = threading.Thread(target=self.threadWorker, daemon=True)
        self.sendThread.start()
        self._write_fut = None

    def threadWorker(self):
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        event_loop.run_until_complete(self.sendWorker())

    async def sendWorker(self):
        while True:
            try:
                content = self.sendBuffer.get()
                if self._write_fut is not None:
                    await self._write_fut
                self._write_fut = asyncio.create_task(self.socket.send(content))
                await self._write_fut
            except Exception as e:
                print(f"发送错误: {e}")
                continue

    def send(self, content: str):
        self.sendBuffer.put(content)

class WebManager(CalculationNodeManager):
    def __init__(self, host='0.0.0.0', port=8765, parallel=False) -> None:
        super().__init__(parallel=parallel)
        self.host = host
        self.port = port

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise exc_val
        self.start(self.host, self.port)

    def start(self, host=None, port=None) -> None:
        if host is None:
            host = self.host
        if port is None:
            port = self.port
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        async def onRecv(socket, path):
            msgIdx = -1
            sendSocket = SendSocket(socket)
            async for message in socket:
                msgIdx += 1
                message = json.loads(message)
                # print(msgIdx, message)

                # key error
                if not all([key in message.keys() for key in ['type', 'params']]):
                    response = json.dumps({
                        'status': 500, 
                        'data': 'no key param'
                    })
                    await socket.send(response)
                
                # key correct
                else:
                    if message['type'] == 'overview':
                        response = json.dumps({
                            'status': 200, 
                            'type': 'overview', 
                            'data': self.inspect()
                        })
                        await socket.send(response)

                    elif message['type'] == 'run':
                        params = message['params']
                        conf = params

                        def image2base64(img):
                            import base64
                            from io import BytesIO
                            from PIL import Image

                            # 创建一个示例NumPy数组（图像）
                            image_np = img

                            # 将NumPy数组转换为PIL.Image对象
                            image_pil = Image.fromarray(image_np)

                            # 将PIL.Image对象保存为字节流
                            buffer = BytesIO()
                            image_pil.save(buffer, format='JPEG')
                            buffer.seek(0)

                            # 使用base64库将字节流编码为base64字符串
                            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                            return image_base64

                        Logger.global_trigger = lambda x: sendSocket.send(json.dumps({
                            'status': 200, 
                            'type': x['type'], 
                            'data': {
                                'content': '[{}]: '.format(x['name']) + ' '.join([str(_) for _ in x['args']]) + getattr(x['kwargs'], 'end', '\n') if x['type'] == 'print' \
                                    else image2base64(x['args'])
                            }
                        }))
                        for logger in Logger.loggers.values():
                            if logger.trigger is None:
                                logger.trigger = Logger.global_trigger

                        try:
                            self.execute(conf)
                        except Exception as e:
                            response = json.dumps({
                                'status': 200, 
                                'type': 'print', 
                                'data': {
                                    'content': traceback.format_exc()
                                }
                            })
                            await socket.send(response)
                    
                    # unknown key
                    else:
                        response = json.dumps({
                            'status': 500, 
                            'data': 'unknown type'
                        })
                        await socket.send(response)

        print('The backend server is running on [{}:{}]...'.format(host, port))
        print('The frontend page is at: https://picpic2013.github.io/dlframe-front/')

        event_loop.run_until_complete(websockets.serve(onRecv, host, port))
        event_loop.run_forever()

    async def handle_message(self, socket, message, sendSocket):
        try:
            message = json.loads(message)
            # 处理消息...
        except Exception as e:
            error_msg = {
                'status': 500,
                'type': 'error',
                'data': {
                    'content': str(e)
                }
            }
            await socket.send(json.dumps(error_msg))
            
    async def onRecv(self, socket, path):
        try:
            sendSocket = SendSocket(socket)
            async for message in socket:
                await self.handle_message(socket, message, sendSocket)
        except websockets.exceptions.ConnectionClosed:
            print("连接已关闭")
        except Exception as e:
            print(f"发生错误: {e}")