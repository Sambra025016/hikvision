"""
hikvision.error
~~~~~~~~~~~~~~~~~~~~

Module errors and exceptionss

Copyright (c) 2015 Finbarr Brady <https://github.com/fbradyirl>
Licensed under the MIT license.

本模块定义海康威视设备通信失败时使用的自定义异常。
"""


class HikvisionError(Exception):

    """
    This exception is raised when there has occurred an error related to
    communication with hikvision. It is a subclass of Exception.

    与海康威视设备通信过程中发生错误时抛出此异常。
    """

    def __init__(self, message='', original=None):
        # 保留高层错误说明以及可选的底层异常，便于调用方排查。
        Exception.__init__(self)
        self.message = message
        self.original = original

    def __str__(self):
        # 若包装了底层异常，则输出其类型和具体错误信息。
        if self.original:
            original_name = type(self.original).__name__
            message = '%s Original exception:'\
                ' %s, "%s"' % (self.message, original_name, str(self.original))
            return message
        return self.message
