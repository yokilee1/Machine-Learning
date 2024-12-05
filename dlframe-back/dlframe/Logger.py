import io
import base64
from PIL import Image

class Logger:
    
    def __init__(self, name, trigger=None) -> None:
        self.name = name
        self.trigger = trigger

    def print(self, *x, **kwargs):
        if self.trigger is not None:
            self.trigger({
                'type': 'print', 
                'name': self.name, 
                'args': x, 
                'kwargs': kwargs
            })
            return
        print('[{}]: '.format(self.name), end=' ')
        print(*x, **kwargs)

    # def imshow(self, img_or_plt):
    #     if isinstance(img_or_plt, Image.Image):
    #         img = img_or_plt
    #     elif hasattr(img_or_plt, 'savefig'):
    #         # 假设 img_or_plt 是 matplotlib.pyplot 对象
    #         buf = io.BytesIO()
    #         img_or_plt.savefig(buf, format='png')
    #         buf.seek(0)
    #         img = Image.open(buf)
    #         buf.close()
    #     else:
    #         raise ValueError("Unsupported image type. Expected PIL.Image.Image or matplotlib.pyplot object.")

    #     if self.trigger is not None:
    #         self.trigger({
    #             'type': 'imshow', 
    #             'name': self.name, 
    #             'args': self.image2base64(img)
    #         })
    #     else:
    #         print(img)

    # @staticmethod
    # def image2base64(image):
    #     buffer = io.BytesIO()
    #     image.save(buffer, format='PNG')
    #     buffer.seek(0)
    #     img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    #     buffer.close()
    #     return img_base64

    def imshow(self, img):
        if self.trigger is not None:
            self.trigger({
                'type': 'imshow', 
                'name': self.name, 
                'args': img
            })
            return
        print(img)

    @classmethod
    def get_logger(cls, name):
        logger = cls(name, trigger=Logger.global_trigger)
        cls.loggers.setdefault(id(logger), logger)
        return logger

Logger.loggers = {}
Logger.global_trigger = None